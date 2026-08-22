import os
import re
import glob
import warnings
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torchvision import transforms as T
import torch

from utils.graphics_utils import focal2fov
from scene.pre_train_pc import get_pointcloud
from scene.endo_loader import CameraInfo


class IMED_Dataset:
    def __init__(
        self,
        datadir,
        downsample=1.0,
        load_test=True,
        use_source_overlap_mask=False,
    ):
        self.root_dir = datadir
        self.downsample = downsample
        self.load_test = load_test
        self.use_source_overlap_mask = use_source_overlap_mask
        self.transform = T.ToTensor()
        self.maxtime = 1.0

        self._validate_structure()
        self.K_map = self._parse_intrinsics()
        self.c2w_map = self._parse_poses()
        self._load_streams()

        self.train_idxs = list(range(len(self.train_records)))
        self.test_idxs = list(range(len(self.test_records)))
        self.video_idxs = self.test_idxs

    def _validate_structure(self):
        assert os.path.isfile(os.path.join(self.root_dir, "pose.txt")), "Missing pose.txt"
        assert os.path.isfile(os.path.join(self.root_dir, "K.txt")), "Missing K.txt"
        scopes = ("endoscope1", "endoscope2") if self.load_test else ("endoscope2",)
        for scope in scopes:
            for subdir in ("L", "depthL", "toolL"):
                path = os.path.join(self.root_dir, scope, subdir)
                assert os.path.isdir(path), f"Missing required folder: {path}"

    def _parse_intrinsics(self):
        k_path = os.path.join(self.root_dir, "K.txt")
        with open(k_path, "r", encoding="utf-8") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        matrices = {}
        i = 0
        while i < len(raw_lines):
            line = raw_lines[i]
            if line.startswith("#"):
                header = line[1:].strip()
                if not header.startswith("K"):
                    i += 1
                    continue
                key = header.split()[0]
                assert i + 3 < len(raw_lines), f"Incomplete matrix block for {key}"
                rows = []
                for j in range(1, 4):
                    vals = [float(v) for v in raw_lines[i + j].split()]
                    assert len(vals) == 3, f"Expected 3 intrinsics values in line: {raw_lines[i + j]}"
                    rows.append(vals)
                matrices[key] = np.array(rows, dtype=np.float32)
                i += 4
                continue
            i += 1

        for needed in ("K1_L", "K2_L"):
            assert needed in matrices, f"Missing intrinsics {needed} in K.txt"
        return matrices

    def _parse_poses(self):
        pose_path = os.path.join(self.root_dir, "pose.txt")
        with open(pose_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) == 2, f"Expected exactly 2 static poses in pose.txt, found {len(lines)}"
        c2w_by_cam = {}
        for line in lines:
            parts = line.split()
            assert len(parts) == 8, f"Pose row must have 8 values: {line}"
            cam_id = int(parts[0])
            t = np.array([float(v) for v in parts[1:4]], dtype=np.float32)
            q = np.array([float(v) for v in parts[4:8]], dtype=np.float32)
            rot = R.from_quat(q).as_matrix().astype(np.float32)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = rot
            c2w[:3, 3] = t
            c2w_by_cam[cam_id] = c2w

        assert 0 in c2w_by_cam and 1 in c2w_by_cam, "pose.txt must provide camera ids 0 and 1"
        return {"cam2": c2w_by_cam[0], "cam1": c2w_by_cam[1]}

    def _extract_frame_id(self, path):
        name = os.path.basename(path)
        match = re.match(r"frame_(\d+)\.(png|npy)$", name)
        assert match is not None, f"Unexpected frame naming: {name}"
        return int(match.group(1))

    def _collect_stream(self, scope):
        rgb_paths = sorted(glob.glob(os.path.join(self.root_dir, scope, "L", "*.png")))
        depth_paths = sorted(glob.glob(os.path.join(self.root_dir, scope, "depthL", "*.npy")))
        mask_paths = sorted(glob.glob(os.path.join(self.root_dir, scope, "toolL", "*.png")))
        assert len(rgb_paths) > 0, f"No RGB frames found in {scope}/L"
        assert len(rgb_paths) == len(depth_paths), f"RGB/depth count mismatch in {scope}"

        masks_missing = len(mask_paths) == 0
        if masks_missing:
            warnings.warn(
                f"No tool masks found in {scope}/toolL; using an all-valid mask in memory.",
                RuntimeWarning,
            )
        else:
            assert len(rgb_paths) == len(mask_paths), f"RGB/mask count mismatch in {scope}"

        rgb_names = [os.path.basename(p).replace(".png", "") for p in rgb_paths]
        depth_names = [os.path.basename(p).replace(".npy", "") for p in depth_paths]
        assert rgb_names == depth_names, f"RGB/depth names mismatch in {scope}"
        if not masks_missing:
            mask_names = [os.path.basename(p).replace(".png", "") for p in mask_paths]
            assert rgb_names == mask_names, f"RGB/mask names mismatch in {scope}"

        records = []
        for i in range(len(rgb_paths)):
            frame_id = self._extract_frame_id(rgb_paths[i])
            records.append(
                {
                    "frame_id": frame_id,
                    "rgb": rgb_paths[i],
                    "depth": depth_paths[i],
                    "mask": None if masks_missing else mask_paths[i],
                }
            )
        return records

    def _load_streams(self):
        self.train_records = self._collect_stream("endoscope2")
        self.test_records = self._collect_stream("endoscope1") if self.load_test else []
        train_ids = [r["frame_id"] for r in self.train_records]
        if self.load_test:
            test_ids = [r["frame_id"] for r in self.test_records]
            assert train_ids == test_ids, "Train/test camera frame ids must align exactly"

        first_img = np.array(Image.open(self.train_records[0]["rgb"]))
        first_depth = np.load(self.train_records[0]["depth"])
        assert first_img.ndim == 3 and first_img.shape[2] == 3, "Expected RGB image with 3 channels"
        assert first_depth.ndim == 2, "Expected depth map as 2D array"
        self.H_rgb, self.W_rgb = int(first_img.shape[0]), int(first_img.shape[1])
        self.H, self.W = int(first_depth.shape[0]), int(first_depth.shape[1])
        assert self.H_rgb % self.H == 0 and self.W_rgb % self.W == 0, "RGB/depth resolution ratio must be integer"
        self.scale_y = self.H_rgb // self.H
        self.scale_x = self.W_rgb // self.W
        assert self.scale_x == self.scale_y, "Anisotropic RGB/depth scaling is not supported"
        assert self.scale_x == 2, "Expected 2x RGB-to-depth scaling for IMED sequence"

    def _load_mask(self, mask_path):
        raw = np.array(Image.open(mask_path))
        if raw.ndim == 3:
            raw = raw[..., 0]
        assert raw.ndim == 2, f"Mask must be 2D at {mask_path}"
        unique_vals = np.unique(raw)
        assert np.all(np.isin(unique_vals, [0, 255])), f"Mask must be binary 0/255 at {mask_path}"
        mask = 1.0 - (raw.astype(np.float32) / 255.0)
        return mask.astype(np.bool_)

    def _load_resized_rgb(self, rgb_path):
        img = Image.open(rgb_path).convert("RGB")
        assert img.size == (self.W_rgb, self.H_rgb), f"RGB shape mismatch at {rgb_path}"
        if (self.W, self.H) != (self.W_rgb, self.H_rgb):
            img = img.resize((self.W, self.H), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        assert arr.shape == (self.H, self.W, 3), f"Resized RGB shape mismatch at {rgb_path}"
        return arr

    def _load_resized_mask(self, mask_path):
        mask = self._load_mask(mask_path).astype(np.uint8) * 255
        img = Image.fromarray(mask, mode="L")
        assert img.size == (self.W_rgb, self.H_rgb), f"Mask shape mismatch at {mask_path}"
        if (self.W, self.H) != (self.W_rgb, self.H_rgb):
            img = img.resize((self.W, self.H), Image.NEAREST)
        out = np.array(img)
        assert out.ndim == 2 and out.shape == (self.H, self.W), f"Resized mask shape mismatch at {mask_path}"
        return (out > 0)

    def _load_record_mask(self, record):
        if record["mask"] is None:
            return np.ones((self.H, self.W), dtype=np.bool_)
        return self._load_resized_mask(record["mask"])

    def _build_source_overlap_mask(self, depth):
        """Return Endoscope2 pixels whose 3D points project into Endoscope1."""
        assert depth.shape == (self.H, self.W), "Depth shape mismatch for overlap mask"

        valid_depth = np.isfinite(depth) & (depth > 0)
        valid_flat = np.flatnonzero(valid_depth.reshape(-1))
        overlap_flat = np.zeros(self.H * self.W, dtype=np.bool_)
        if valid_flat.size == 0:
            return overlap_flat.reshape(self.H, self.W)

        source_v, source_u = np.divmod(valid_flat, self.W)
        z2 = depth.reshape(-1)[valid_flat].astype(np.float64)

        K2 = self.K_map["K2_L"].astype(np.float64).copy()
        K1 = self.K_map["K1_L"].astype(np.float64).copy()
        K2[0, :] /= self.downsample * self.scale_x
        K2[1, :] /= self.downsample * self.scale_y
        K1[0, :] /= self.downsample * self.scale_x
        K1[1, :] /= self.downsample * self.scale_y

        x2 = (source_u.astype(np.float64) - K2[0, 2]) * z2 / K2[0, 0]
        y2 = (source_v.astype(np.float64) - K2[1, 2]) * z2 / K2[1, 1]
        points_cam2 = np.stack((x2, y2, z2, np.ones_like(z2)), axis=1)

        cam2_to_cam1 = np.linalg.inv(self.c2w_map["cam1"]) @ self.c2w_map["cam2"]
        points_cam1 = (cam2_to_cam1.astype(np.float64) @ points_cam2.T).T[:, :3]
        z1 = points_cam1[:, 2]
        in_front = np.isfinite(points_cam1).all(axis=1) & (z1 > 1e-6)

        target_u = np.full(z1.shape, -1, dtype=np.int64)
        target_v = np.full(z1.shape, -1, dtype=np.int64)
        target_u[in_front] = np.rint(
            K1[0, 0] * points_cam1[in_front, 0] / z1[in_front] + K1[0, 2]
        ).astype(np.int64)
        target_v[in_front] = np.rint(
            K1[1, 1] * points_cam1[in_front, 1] / z1[in_front] + K1[1, 2]
        ).astype(np.int64)
        inside = (
            in_front
            & (target_u >= 0)
            & (target_u < self.W)
            & (target_v >= 0)
            & (target_v < self.H)
        )
        overlap_flat[valid_flat[inside]] = True
        return overlap_flat.reshape(self.H, self.W)

    def _record_to_camera(self, record, c2w, K, time_val, uid, apply_source_overlap=False):
        color = self._load_resized_rgb(record["rgb"])
        depth = np.load(record["depth"]).astype(np.float32)
        assert depth.shape == (self.H, self.W), f"Depth shape mismatch at {record['depth']}"
        mask = self._load_record_mask(record)
        assert mask.shape == (self.H, self.W), f"Mask shape mismatch for {record['rgb']}"
        if apply_source_overlap:
            overlap_mask = self._build_source_overlap_mask(depth)
            tool_free_fraction = float(mask.mean())
            source_visible_fraction = float(overlap_mask.mean())
            mask = mask & overlap_mask
            self._source_overlap_stats.append(
                (tool_free_fraction, source_visible_fraction, float(mask.mean()))
            )

        fx = float(K[0, 0]) / (self.downsample * self.scale_x)
        fy = float(K[1, 1]) / (self.downsample * self.scale_y)
        cx = float(K[0, 2]) / (self.downsample * self.scale_x)
        cy = float(K[1, 2]) / (self.downsample * self.scale_y)

        w2c = np.linalg.inv(c2w)
        rot = w2c[:3, :3]
        trans = w2c[:3, 3]
        rot = np.transpose(rot)

        image = self.transform(np.ascontiguousarray(color))
        depth_t = torch.from_numpy(np.ascontiguousarray(depth[None, ...]))
        mask_t = torch.from_numpy(np.ascontiguousarray(mask))
        fov_x = focal2fov(fx, self.W)
        fov_y = focal2fov(fy, self.H)
        return CameraInfo(
            uid=uid,
            R=rot,
            T=trans,
            FovY=fov_y,
            FovX=fov_x,
            image=image,
            depth=depth_t,
            image_path=record["rgb"],
            image_name=os.path.basename(record["rgb"]),
            width=self.W,
            height=self.H,
            time=time_val,
            mask=mask_t,
            Zfar=None,
            Znear=None,
            pc=None,
        )

    def format_infos(self, split):
        assert split in ("train", "test", "video"), f"Unsupported split: {split}"
        if split == "train":
            records = self.train_records
            c2w = self.c2w_map["cam2"]
            K = self.K_map["K2_L"]
        else:
            records = self.test_records
            c2w = self.c2w_map["cam1"]
            K = self.K_map["K1_L"]

        apply_source_overlap = split == "train" and self.use_source_overlap_mask
        self._source_overlap_stats = []
        n = len(records)
        denom = max(n - 1, 1)
        cams = []
        for idx, record in enumerate(records):
            time_val = idx / denom
            cams.append(
                self._record_to_camera(
                    record,
                    c2w,
                    K,
                    time_val,
                    idx,
                    apply_source_overlap=apply_source_overlap,
                )
            )
        if apply_source_overlap and self._source_overlap_stats:
            stats = np.asarray(self._source_overlap_stats, dtype=np.float64)
            print(
                "iMED source-overlap training mask: "
                f"frames={len(stats)}, "
                f"tool_free_mean={stats[:, 0].mean():.4f}, "
                f"source_visible_mean={stats[:, 1].mean():.4f}, "
                f"effective_mean={stats[:, 2].mean():.4f}"
            )
        return cams

    def get_pretrain_pcd(self):
        first = self.train_records[0]
        color = self._load_resized_rgb(first["rgb"]).transpose(2, 0, 1)
        depth = np.load(first["depth"]).astype(np.float32)[None, ...]
        mask = self._load_record_mask(first)[None, ...]

        K = self.K_map["K2_L"]
        intrinsics = [
            float(K[0, 0]) / self.scale_x,
            float(K[1, 1]) / self.scale_y,
            float(K[0, 2]) / self.scale_x,
            float(K[1, 2]) / self.scale_y,
        ]
        w2c = np.linalg.inv(self.c2w_map["cam2"])
        pts, cols = get_pointcloud(color, depth, intrinsics, w2c, mask=mask)
        assert pts.shape[0] > 0, "Pretrain point cloud is empty"
        normals = np.zeros((pts.shape[0], 3), dtype=np.float32)
        return pts, cols, normals

    def get_maxtime(self):
        return self.maxtime
