# Observed Phase 1 baseline

Actual run: 2026-11-07T04:17:07.036875+00:00. Five warmups and 30 measured repetitions per timing pass; one OpenCV CPU thread. Seed 2026. Input resolutions shown below are resized to half width/height.

| Input | Batch | CPU median ms/batch | CPU p95 ms/batch | Frames/s | Peak process RSS MiB |
|---|---:|---:|---:|---:|---:|
| 480p | 1 | 3.197 | 3.289 | 311.8 | 68.9 |
| 480p | 4 | 12.755 | 12.901 | 313.2 | 87.2 |
| 480p | 8 | 25.639 | 25.773 | 312.2 | 111.9 |
| 720p | 1 | 9.217 | 9.277 | 108.5 | 79.0 |
| 720p | 4 | 50.080 | 50.411 | 79.7 | 135.5 |
| 720p | 8 | 100.824 | 102.418 | 79.1 | 209.5 |
| 1080p | 1 | 21.077 | 21.689 | 47.3 | 104.1 |
| 1080p | 4 | 108.200 | 108.404 | 37.0 | 227.8 |
| 1080p | 8 | 224.371 | 227.419 | 35.6 | 394.0 |

CPU stage profile medians (separate synchronized profiling pass):

| Input | Batch | Preprocess ms | Resize ms | Grayscale ms | Gaussian ms |
|---|---:|---:|---:|---:|---:|
| 480p | 1 | 1.264 | 1.225 | 0.408 | 0.256 |
| 480p | 4 | 5.012 | 5.142 | 1.577 | 0.905 |
| 480p | 8 | 9.926 | 10.290 | 3.487 | 1.802 |
| 720p | 1 | 3.763 | 3.646 | 1.112 | 0.635 |
| 720p | 4 | 21.548 | 19.887 | 5.434 | 2.647 |
| 720p | 8 | 42.087 | 40.607 | 11.666 | 5.525 |
| 1080p | 1 | 8.399 | 8.350 | 2.866 | 1.414 |
| 1080p | 4 | 46.723 | 41.963 | 13.239 | 5.866 |
| 1080p | 8 | 93.402 | 84.784 | 30.758 | 14.223 |

All nine CPU cases completed. All GPU cases are explicitly unavailable; CUDA parity, GPU memory, stage acceleration and CPU/GPU speedups remain unmeasured. The report contains no GPU timings. Read `cuda-diagnostics.log` together with the report warnings: CUDA initialization reports driver version 12020 versus the installed CUDA 13.2 PyTorch build, while NVIDIA memory-manager initialization also fails. No drivers or dependencies were changed.

The CPU preprocessing stage converts the full input batch to normalized float32 before resize, so it incurs full-resolution memory traffic and allocations. Its measured time is visible separately above; grayscale and blur operate on quarter as many output pixels. These observations describe the CPU implementation and cannot establish which stages would accelerate on GPU. Host wrapping/output formatting remain CPU work in the GPU implementation; device conversion, memory transfers and synchronization add costs that only a working CUDA run can quantify.

Use the common stage speedup ratios to identify actual compute benefits on working CUDA. Use end-to-end speedup to decide whether those benefits survive transfer and conversion overhead. A value at or below 1 means no net GPU benefit for that workload; no such conclusion can be drawn from this CPU-only baseline.

Absolute process peak RSS includes interpreter/imports, input, warmups and stage profiling. It is not isolated incremental algorithm RAM. This run used the current system clocks and power settings without controlling thermal state or competing processes; it is an observed baseline, not a controlled hardware performance claim. See the JSON for raw samples, hardware/software, source hashes, workload hashes, and configuration.
