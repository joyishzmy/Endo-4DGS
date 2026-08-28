#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import faulthandler
faulthandler.enable()
import numpy as np
import random
import os, sys
import torch
import cv2
from random import randint
import traceback
from utils.loss_utils import l1_loss, l2_loss, robust_log_depth_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, build_rotation
import uuid
from tqdm import tqdm
import torch.nn.functional as F
from utils.image_utils import psnr, ssim, lpips_score
from metrics import _build_global_imed_overlap_mask, masked_psnr, masked_ssim
from utils.loss_utils import mae_loss
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
from utils.scene_utils import render_training_image
from utils.loss_utils import GradL1Loss, confidence_loss, TV_loss
from utils.graphics_utils import get_pseudo_normal
from utils.imed_overlap import (
    blend_imed_soft_overlap_weight,
    build_imed_soft_overlap_weight,
    imed_soft_overlap_anneal_alpha,
)
from utils.imed_stereo import use_stereo_auxiliary_for_stage
from time import time
import copy
import open3d as o3d


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



def scene_reconstruction(mp, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer):
    first_iter = 0
    use_depth = pipe.use_depth
    use_smooth = pipe.use_smooth
    use_normal = pipe.use_normal
    use_confidence = pipe.use_confidence
    print('Init with pretrain:', mp.use_pretrain)
    print('Use depth l1:', use_depth)
    print('Use smooth:', use_smooth)
    print('Use normal:', use_normal)
    print('Use confidence:', use_confidence)
    
    gaussians.training_setup(opt)
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if mp.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    viewpoint_stack = None
    final_iter = train_iter
    soft_overlap_logged = False
    stereo_gate_logged = False
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training")
    first_iter += 1
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    if mp.imed_use_principal_point:
        if any(
            getattr(camera_info, "cx", None) is None
            or getattr(camera_info, "cy", None) is None
            for camera_info in train_cams.dataset
        ):
            raise ValueError("Principal-point mode requires cx/cy for every training camera")
        print("iMED off-center principal-point projection enabled")

    stereo_pairs = None
    stereo_aux_active = False
    if mp.imed_use_stereo:
        if mp.imed_stereo_right_appearance_only and not mp.imed_stereo_photometric_gate:
            raise ValueError("Appearance-only stereo requires photometric gating")
        assert not opt.dataloader, "iMED paired stereo currently requires dataloader=False"
        assert not opt.zerostamp_init, "iMED paired stereo is incompatible with zerostamp_init"
        assert opt.batch_size == 2, "iMED paired stereo requires batch_size=2 (one L/R pair)"
        pair_map = {}
        for camera_index, camera_info in enumerate(train_cams.dataset):
            pair_id = getattr(camera_info, "stereo_pair_id", -1)
            eye = getattr(camera_info, "stereo_eye", "mono")
            if pair_id < 0 or eye not in ("L", "R"):
                raise ValueError("Stereo training camera is missing a valid pair id/eye")
            pair_map.setdefault(pair_id, {})[eye] = camera_index
        if not pair_map or any(set(pair) != {"L", "R"} for pair in pair_map.values()):
            raise ValueError("Every iMED stereo timestamp must contain exactly one L and one R camera")
        stereo_pairs = [(pair["L"], pair["R"]) for _, pair in sorted(pair_map.items())]
        if mp.imed_stereo_right_rgb_only:
            for left_index, right_index in stereo_pairs:
                if not getattr(train_cams.dataset[left_index], "depth_supervision", True):
                    raise ValueError("Left stereo cameras must retain depth supervision")
                if getattr(train_cams.dataset[right_index], "depth_supervision", True):
                    raise ValueError("RGB-only mode must disable right pseudo-depth supervision")
            print("iMED right camera supervision: RGB only; pseudo-depth is visibility-only")
        if mp.imed_stereo_photometric_gate:
            if not mp.imed_stereo_right_rgb_only:
                raise ValueError("Photometric gating requires right RGB-only supervision")
            if not (0.0 <= mp.imed_stereo_right_loss_weight <= 1.0):
                raise ValueError("Right stereo auxiliary loss weight must be in [0, 1]")
            for left_index, right_index in stereo_pairs:
                if getattr(train_cams.dataset[left_index], "stereo_rgb_weight", None) is not None:
                    raise ValueError("Left cameras must not have a stereo photometric gate")
                if getattr(train_cams.dataset[right_index], "stereo_rgb_weight", None) is None:
                    raise ValueError("Right cameras are missing stereo photometric gates")
            print(
                "iMED right camera loss: detached photometric gate, "
                f"aux_weight={mp.imed_stereo_right_loss_weight:.4f}"
            )
            if mp.imed_stereo_right_appearance_only:
                print(
                    "iMED right camera gradients: SH appearance only; "
                    "geometry/deformation/densification isolated"
                )
            stereo_aux_active = use_stereo_auxiliary_for_stage(
                stage, mp.imed_stereo_right_fine_only
            )
            if mp.imed_stereo_right_fine_only:
                print(
                    "iMED staged stereo auxiliary: "
                    f"stage={stage}, active={stereo_aux_active}"
                )
        else:
            stereo_aux_active = True
        effective_batch_size = 2 if stereo_aux_active else 1
        print(
            f"iMED paired stereo sampler: pairs={len(stereo_pairs)}, "
            f"effective_batch_size={effective_batch_size}"
        )

    if not viewpoint_stack and not opt.dataloader and stereo_pairs is None:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
    # op dataloader: False
    batch_size = opt.batch_size
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,sampler=sampler,num_workers=32,collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=32,collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)
    
    
    # dynerf, zerostamp_init
    # breakpoint()
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack,0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False 
    
    count = 0
    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, ts = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage, \
                        cam_type=scene.dataset_type)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, mp.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # dynerf's branch
        if stereo_pairs is not None:
            left_index, right_index = random.choice(stereo_pairs)
            viewpoint_cams = [train_cams[left_index]]
            if stereo_aux_active:
                viewpoint_cams.append(train_cams[right_index])
        elif opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)
        else:
            idx = 0
            viewpoint_cams = []

            while idx < batch_size :    
                viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
                if not viewpoint_stack :
                    viewpoint_stack =  temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1
            if len(viewpoint_cams) == 0:
                continue
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        # read depth
        gt_depths = []
        depths = []
        masks = []
        depth_supervision_masks = []
        stereo_rgb_weights = []
        source_overlap_masks = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        appearance_only_view_flags = []
        gs_normal = []
        if use_confidence:
            confidences = []
        # ranking_loss = EdgeguidedRankingLoss(point_pairs=5000, alpha=0.8)
        # pc_loss = PearsonCorrCoef().cuda()
        # silog_loss = SILogLoss()
        grad_loss = GradL1Loss()
        avg_filter = torch.nn.AvgPool2d(2)
        jump = False
        # SSI_loss = ScaleAndShiftInvariantLoss()
        for viewpoint_cam in viewpoint_cams:
            # pc = viewpoint_cam.pc.cuda()
            # src_pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(gaussians.get_xyz.detach().cpu().numpy(), o3d.core.float32))
            # tgt_pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(pc.detach().cpu().numpy(), o3d.core.float32))
            # result = o3d.t.pipelines.registration.icp(src_pcd, tgt_pcd, 5)
            # corr_set = result.correspondence_set.numpy()
            appearance_only_view = bool(
                stereo_aux_active
                and mp.imed_stereo_right_appearance_only
                and getattr(viewpoint_cam, "stereo_eye", "mono") == "R"
            )
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage, \
                cam_type=scene.dataset_type, iteration=count,
                appearance_only=appearance_only_view)
            image, viewspace_point_tensor, radii, depth = \
                render_pkg["render"], render_pkg["viewspace_points"], \
                render_pkg["radii"], render_pkg['depth']
            visibility_filter = radii>0
            gt_depth = viewpoint_cam.depth.cuda()
            mask = viewpoint_cam.mask

            if scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()

            if mask is not None:
                mask = mask.cuda()
                masks.append(mask.unsqueeze(0))
                if getattr(viewpoint_cam, "depth_supervision", True):
                    depth_supervision_masks.append(mask.unsqueeze(0))
                else:
                    depth_supervision_masks.append(torch.zeros_like(mask).unsqueeze(0))
                if stereo_aux_active and mp.imed_stereo_photometric_gate:
                    stereo_rgb_weight = getattr(viewpoint_cam, "stereo_rgb_weight", None)
                    if stereo_rgb_weight is None:
                        stereo_rgb_weight = torch.ones_like(mask, dtype=torch.float32)
                    else:
                        stereo_rgb_weight = stereo_rgb_weight.cuda().float()
                    stereo_rgb_weights.append(stereo_rgb_weight.unsqueeze(0))
            source_overlap_mask = getattr(viewpoint_cam, "source_overlap_mask", None)
            if source_overlap_mask is not None:
                source_overlap_masks.append(source_overlap_mask.cuda().unsqueeze(0))
            
            images.append(image.unsqueeze(0))
            dep_mask = torch.logical_and(gt_depth > 0, depth > 0)
            gt_depth = gt_depth * dep_mask
            depth = depth * dep_mask
            depths.append(depth.unsqueeze(0))
            if use_normal:
                gs_normal.append(render_pkg['normal'].unsqueeze(0))
            if use_confidence:
                confidences.append(render_pkg['confidence'].unsqueeze(0))
                
            gt_depths.append(gt_depth.unsqueeze(0))
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            appearance_only_view_flags.append(appearance_only_view)

        geometry_view_indices = [
            index for index, flag in enumerate(appearance_only_view_flags) if not flag
        ]
        if not geometry_view_indices:
            raise ValueError("At least one primary geometry view is required")
        radii = torch.cat([radii_list[index] for index in geometry_view_indices], 0).max(dim=0).values
        visibility_filter = torch.cat(
            [visibility_filter_list[index] for index in geometry_view_indices], 0
        ).any(dim=0)
        
        if len(masks) != 0:
            mask_tensor = torch.cat(masks, 0)
            depth_supervision_tensor = torch.cat(depth_supervision_masks, 0)
            stereo_rgb_weight_tensor = (
                torch.cat(stereo_rgb_weights, 0)
                if stereo_rgb_weights
                else None
            )
        else:
            mask_tensor = None
            depth_supervision_tensor = None
            stereo_rgb_weight_tensor = None
        if source_overlap_masks:
            assert len(source_overlap_masks) == len(viewpoint_cams), (
                "Soft source-overlap masks must be present for every camera in a batch"
            )
            source_overlap_tensor = torch.cat(source_overlap_masks, 0)
            soft_overlap = build_imed_soft_overlap_weight(
                mask_tensor,
                source_overlap_tensor,
                mp.imed_source_overlap_visibility_threshold,
            )
            soft_overlap_alpha = imed_soft_overlap_anneal_alpha(
                stage,
                iteration,
                final_iter,
                mp.imed_source_overlap_anneal_start,
            )
            rgb_loss_weight = blend_imed_soft_overlap_weight(
                mask_tensor,
                soft_overlap.weight,
                soft_overlap_alpha,
            )
            if not soft_overlap_logged:
                valid_weights = rgb_loss_weight[mask_tensor.bool()]
                sum_ratio = rgb_loss_weight.sum() / mask_tensor.float().sum().clamp_min(1.0)
                active_count = soft_overlap.active.sum().item()
                frame_count = soft_overlap.active.numel()
                print(
                    "iMED soft-overlap RGB weights: "
                    f"visibility_threshold={mp.imed_source_overlap_visibility_threshold:.4f}, "
                    f"raw_visibility_min={soft_overlap.raw_visibility.min().item():.4f}, "
                    f"raw_visibility_max={soft_overlap.raw_visibility.max().item():.4f}, "
                    f"active_frames={active_count}/{frame_count}, "
                    f"anneal_alpha={soft_overlap_alpha:.4f}, "
                    f"min={valid_weights.min().item():.4f}, "
                    f"max={valid_weights.max().item():.4f}, "
                    f"sum_ratio={sum_ratio.item():.4f}"
                )
                soft_overlap_logged = True
            if (
                stage == "fine"
                and mp.imed_source_overlap_anneal_start >= 0
                and iteration in {
                    mp.imed_source_overlap_anneal_start,
                    (mp.imed_source_overlap_anneal_start + final_iter) // 2,
                    final_iter - 1,
                }
            ):
                print(
                    "iMED soft-overlap late annealing: "
                    f"iteration={iteration}/{final_iter}, "
                    f"alpha={soft_overlap_alpha:.4f}"
                )
        else:
            source_overlap_tensor = None
            rgb_loss_weight = mask_tensor
        
        valid_mask = mask_tensor.unsqueeze(1)
        depth_valid_mask = depth_supervision_tensor.unsqueeze(1)
        image_tensor = torch.cat(images,0) * valid_mask
        depth_tensor = torch.cat(depths, 0) * depth_valid_mask
        gt_image_tensor = torch.cat(gt_images,0) * valid_mask
        gt_depth_tensor = torch.cat(gt_depths, 0) * depth_valid_mask
        
        if use_normal:
            gs_normal = torch.cat(gs_normal, 0) * valid_mask
        if use_confidence:
            confidences = torch.cat(confidences, 0) * valid_mask

        # In G3/G4 the synchronized right view is an auxiliary appearance cue.
        # All original RGB-D/normal/confidence regularizers retain the exact
        # left-view batch semantics of the principal-point baseline.
        primary_slice = slice(0, 1) if stereo_aux_active and mp.imed_stereo_photometric_gate else slice(None)
        primary_depth = depth_tensor[primary_slice]
        primary_gt_depth = gt_depth_tensor[primary_slice]
        primary_depth_mask = depth_valid_mask[primary_slice]
        
        # Loss
        if stereo_aux_active and mp.imed_stereo_photometric_gate:
            if len(viewpoint_cams) != 2 or [cam.stereo_eye for cam in viewpoint_cams] != ["L", "R"]:
                raise ValueError("Photometric stereo loss requires one ordered L/R pair")
            left_rgb_loss = l1_loss(
                image_tensor[:1],
                gt_image_tensor[:1],
                rgb_loss_weight[:1].unsqueeze(1),
            )
            right_rgb_loss_weight = (
                rgb_loss_weight[1:2] * stereo_rgb_weight_tensor[1:2]
            )
            right_rgb_loss = l1_loss(
                image_tensor[1:2],
                gt_image_tensor[1:2],
                right_rgb_loss_weight.unsqueeze(1),
            )
            Ll1 = left_rgb_loss + mp.imed_stereo_right_loss_weight * right_rgb_loss
            if not stereo_gate_logged:
                right_valid = mask_tensor[1:2].bool()
                valid_gate = stereo_rgb_weight_tensor[1:2][right_valid]
                if valid_gate.numel() == 0:
                    raise ValueError("Right stereo view has no geometrically valid RGB pixels")
                print(
                    "iMED detached right RGB gate: "
                    f"mean={valid_gate.mean().item():.4f}, "
                    f"min={valid_gate.min().item():.4f}, "
                    f"max={valid_gate.max().item():.4f}"
                )
                stereo_gate_logged = True
        else:
            Ll1 = l1_loss(image_tensor, gt_image_tensor, rgb_loss_weight.unsqueeze(1))
        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # norm
        if use_depth:
            depth_weight = hyper.depth_weight
            if mp.imed_metric_depth_loss:
                depth_loss = robust_log_depth_loss(
                    primary_depth, primary_gt_depth, primary_depth_mask
                ) * depth_weight
            else:
                depth_loss = l1_loss(primary_depth/(primary_depth.max()+1e-6), primary_gt_depth/(primary_gt_depth.max()+1e-6), \
                    mask=primary_depth_mask)*depth_weight
            loss = Ll1 + depth_loss 
        else:
            loss = Ll1
        
        if use_smooth:
            grad_weight=hyper.depth_weight
            sm_loss = (grad_loss(
                primary_depth,
                primary_gt_depth,
                mask=depth_supervision_tensor[primary_slice],
            )) * grad_weight
            loss += sm_loss
            
        if use_normal:
            normal_weight = hyper.normal_weight
            pseudo_normal=get_pseudo_normal(primary_gt_depth, primary_depth_mask)
            primary_gs_normal = gs_normal[primary_slice]
            pseudo_normal = F.interpolate(pseudo_normal, primary_gs_normal.shape[2:4])
            normal_loss = mae_loss(primary_gs_normal, pseudo_normal, primary_depth_mask)*normal_weight
            loss += normal_loss
            
        if use_confidence:
            un_img_weight = hyper.un_img_weight
            un_dep_weight = hyper.un_dep_weight
            confidence_image_mask = valid_mask
            if stereo_aux_active and mp.imed_stereo_photometric_gate:
                confidence_image_mask = valid_mask.clone()
                confidence_image_mask[1:] = False
            confidence_loss_img = confidence_loss(gt_image_tensor, image_tensor, \
                confidences, confidence_image_mask)*un_img_weight
            primary_confidences = confidences[primary_slice]
            confidence_loss_dep = confidence_loss(primary_gt_depth/primary_gt_depth.max(), \
                primary_depth/primary_depth.max(), primary_confidences, primary_depth_mask)*un_dep_weight
            loss += confidence_loss_img
            loss += confidence_loss_dep
            
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            tv_image_tensor = image_tensor[primary_slice]
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, \
                hyper.l1_time_planes, hyper.plane_tv_weight) + \
                    +(TV_loss(primary_depth)+TV_loss(tv_image_tensor))*hyper.depth_weight
            loss += tv_loss
            
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor, gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
            
        loss.backward()
            
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            loss_dict = {"Loss": f"{Ll1.item():.{4}f}",
                        "psnr": f"{psnr_:.{2}f}"}
            if stage == "fine" and hyper.time_smoothness_weight != 0:
                loss_dict['tv_loss'] = f"{tv_loss:.{4}f}"
            if use_depth:
                loss_dict['depth'] = f"{depth_loss:.{4}f}"
            if use_smooth:
                loss_dict["Smooth"] = f"{sm_loss:.{4}f}"
            if opt.lambda_dssim != 0:
                loss_dict["ssim"] = f"{ssim_loss:.{4}f}"
            if use_normal:
                loss_dict["Norm"] = f"{normal_loss:.{4}f}"
            if use_confidence:
                loss_dict["Un_img"] = f"{confidence_loss_img:.{4}f}"
                loss_dict["Un_dep"] = f"{confidence_loss_dep:.{4}f}"
            print(loss_dict)
            
            os.execv(sys.executable, [sys.executable] + sys.argv)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in geometry_view_indices:
            if jump:
                viewspace_point_tensor_grad = viewspace_point_tensor_grad
            else:
                viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # Progress bar
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                string_dict = {"Loss": f"{Ll1.item():.{4}f}",
                                        "psnr": f"{psnr_:.{2}f}"}
                if stage == 'fine':
                    string_dict['tv'] = f"{tv_loss:.{4}f}"
                if use_depth:
                    string_dict["Dep"] = f"{depth_loss:.{4}f}"
                if use_smooth:
                    string_dict["Sm"] = f"{sm_loss:.{4}f}"
                if use_normal:
                    string_dict["Norm"] = f"{normal_loss:.{4}f}"
                if use_confidence:
                    string_dict["Un_img"] = f"{confidence_loss_img:.{4}f}"
                    string_dict["Un_dep"] = f"{confidence_loss_dep:.{4}f}"
                    
                progress_bar.set_postfix(string_dict)
                    
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), \
                testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if mp.render_process and len(test_cams) > 0:
                if (iteration < 1000 and iteration % 10 == 9) \
                    or (iteration < 3000 and iteration % 50 == 49) \
                        or (iteration < 60000 and iteration %  100 == 99) :
                        render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], \
                            render, pipe, background, stage+"test", iteration,timer.get_elapsed_time(),scene.dataset_type)
            timer.start()
            
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - \
                        opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init \
                        - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  
                
                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000:
                    # print('Densify')
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                    # gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000:
                    # print('Prune')
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000 and opt.add_point:
                    # print('Grow')
                    gaussians.grow(5,5,scene.model_path,iteration,stage)
                    # torch.cuda.empty_cache()
                if iteration % opt.opacity_reset_interval == 0:
                    # print("reset opacity")
                    gaussians.reset_opacity()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")

def training(model_param, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(model_param.sh_degree, hyper)
    model_param.model_path = args.model_path
    timer = Timer()
    # The iMED challenge forbids loading Endoscope 1 frames during training.
    # Rendering creates its own Scene with the default load_test_cameras=True.
    scene = Scene(model_param, gaussians, load_coarse=None, load_test_cameras=False)
    timer.start()
    scene_reconstruction(model_param, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer)
    scene_reconstruction(model_param, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations,timer)

def prepare_output_and_logger(expname):    
    if not args.model_path:
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage, dataset_type):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 
        validation_configs = ({'name': 'test', 'cameras': list(scene.getTestCameras())},)
        use_imed_overlap_eval = "imed" in args.source_path.lower()

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_score_test = 0.0
                ssim_test = 0.0
                overlap_mask = None

                if use_imed_overlap_eval:
                    sample_image = config['cameras'][0].original_image
                    out_h, out_w = int(sample_image.shape[-2]), int(sample_image.shape[-1])
                    overlap_np = _build_global_imed_overlap_mask(args.source_path, out_h, out_w)
                    overlap_mask = torch.from_numpy(overlap_np).unsqueeze(0).unsqueeze(0).to(
                        device="cuda", dtype=torch.float32
                    )

                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    pred = image.unsqueeze(0)
                    gt = gt_image.unsqueeze(0)
                    mask = viewpoint.mask
                    if mask is None:
                        frame_mask = torch.ones(
                            (1, 1, image.shape[-2], image.shape[-1]),
                            device="cuda", dtype=torch.float32
                        )
                    else:
                        frame_mask = mask.to(device="cuda", dtype=torch.float32)
                        if frame_mask.ndim == 2:
                            frame_mask = frame_mask.unsqueeze(0).unsqueeze(0)
                        elif frame_mask.ndim == 3:
                            frame_mask = frame_mask.unsqueeze(0)
                        frame_mask = (frame_mask > 0.5).float()
                    if overlap_mask is not None:
                        frame_mask = frame_mask * overlap_mask

                    mask3 = frame_mask.expand(-1, pred.shape[1], -1, -1)
                    valid_values = mask3.sum().clamp_min(1.0)
                    l1_test += ((pred - gt).abs() * mask3).sum().double() / valid_values
                    psnr_test += masked_psnr(pred, gt, frame_mask).double()
                    ssim_test += masked_ssim(pred, gt, frame_mask).double()
                    lpips_score_test += lpips_score(pred * frame_mask, gt * frame_mask).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])  
                lpips_score_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])  

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} lpips_score {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_score_test))
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - lpips_score', lpips_score_test, iteration)


        if tb_writer:
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
        
        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500*i for i in range(100)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 3000, 4000, 5000, 6000, 7_000, 9000, 10000, 12000, 14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--seed", type=int, default=0)
    
    
    args = parser.parse_args(sys.argv[1:])
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.seed)
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)
    # All done
    print("\nTraining complete.")
