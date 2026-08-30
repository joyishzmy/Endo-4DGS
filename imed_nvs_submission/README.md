# iMED NVS final Docker

This package implements the official Task 2 contract described in
`iMED_NVS_Submission_Guidelines.pdf`:

- input is mounted read-only at `/input`;
- output is mounted writable at `/output`;
- only `endoscope2`, `K.txt`, and `pose.txt` are used;
- predictions are RGB PNGs under `/output/<sequence>/renders/`;
- the callable entry point is `new_view()` in `predict.py`;
- the executable container entry point invokes that function for all sequences.

The frozen method is normalized soft-overlap Endo-4DGS with seed 1 and 2500
fine iterations. Training and model files live under `/tmp` and are deleted
after each sequence so the final output contains predictions only.
Training-time metric evaluation is disabled; it is unnecessary for inference
and avoids any runtime download of perceptual-network weights in the official
network-isolated evaluator.

Build on x86-64 Linux:

```bash
git submodule update --init submodules/diff-gaussian-rasterization-depth
docker build --platform linux/amd64 -t imed-nvs-final:dev .
```

Run the official-style local test on a small input directory first:

```bash
./scripts/local_test.sh imed-nvs-final:dev /path/to/input /path/to/output
```
