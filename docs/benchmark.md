# Phase 1: CPU versus CUDA preprocessing benchmark (`cvbench`)

> Moved from the top-level README when the project refocused on the measurement pipeline. Commands and inline paths are relative to the repository root; later measurements are linked below.
>
> **Update:** GPU timings have since been measured with JetPack's system PyTorch 2.1 / CUDA 12.2 (`python3 -m cvbench --backend both`), see [`results/gpu_system_python.json`](../results/gpu_system_python.json) and [the measured evaluation](evaluation.md#performance). The text below describes the original CPU-only baseline from the `.venv`, whose PyTorch cannot use CUDA.

Two independently executable implementations process identical seeded RGB frames:

```bash
.venv/bin/python -m cvbench --backend cpu --output results/cpu.json
.venv/bin/python -m cvbench --backend gpu --output results/gpu.json
.venv/bin/python -m cvbench --backend both --output results/comparison.json
.venv/bin/python -m pytest -q
```

Core dependencies are listed in `requirements.txt`; PyTorch and optional psutil are in `requirements-gpu.txt`. Use an existing compatible environment; this project does not install drivers or modify dependencies. On Jetson, PyTorch must be compatible with the installed JetPack and driver. The checked-in baseline records the exact installed package versions and source hashes. To reproduce a workload, use its recorded configuration, seed, source version/hashes, software, and hardware. Timings also depend on power mode, clocks, thermals, and competing processes; close competing workloads and record the active `nvpmodel -q` and `jetson_clocks --show` outputs alongside a controlled Jetson run. The report records the power configuration file, readable temperatures and CPU clocks, but these do not establish the active power mode or lock clocks.

Default matrix: 640×480, 1280×720, and 1920×1080 **input** resolutions, each at batches 1, 4, and 8. Each input is resized to half width and height. Workload generation and checksums are outside timers. The seeded synthetic noise exercises numerical differences and processing costs; it is not a representative camera scene or a detection accuracy dataset. There are no measurement detection or video AI models in Phase 1.

```bash
# Quick smoke run; defaults use 5 warmups and 30 measured repetitions.
.venv/bin/python -m cvbench --resolutions 480p --batches 1 --warmup 2 --repetitions 5 --output /tmp/smoke.json
```

`CPUBackend(Config(...)).run(frames)` and `GPUBackend(Config(...)).run(frames)` also expose the processing paths directly. CUDA construction fails explicitly when unavailable; the harness records `unavailable` and leaves comparisons null. CPU processing uses NumPy/OpenCV and never falls back to GPU. OpenCV OpenCL is disabled in benchmark workers; OpenCV defaults to one CPU thread, configurable with `--threads`.

The shared numerical contract is uint8 NHWC RGB input, conversion to float32 [0,1], bilinear resize with half-pixel coordinates and no antialiasing, grayscale weights (0.299, 0.587, 0.114), then separable Gaussian blur (5 taps, sigma 1.2, reflect101 boundaries). Both implementations use the same explicit float32 Gaussian coefficients from OpenCV. Output is contiguous float32 NHW grayscale in [0,1]. CUDA uses PyTorch interpolation and two separable convolutions, with TF32 disabled and deterministic cuDNN selection. Small floating-point differences are expected; final output parity is tested at absolute tolerance 2e-5 with zero relative tolerance. CUDA tests cover random, constant and edge patterns, non-integer resize geometry and batches 1/4/8, and skip when CUDA is unavailable.

Each backend/case executes in a fresh subprocess. Setup and kernel construction are outside timings. Five full warmup iterations precede 30 full end-to-end repetitions; another 30 repetitions collect stage profiles. Both distributions retain raw samples and report median and p95 (NumPy linear percentiles). End-to-end wall-clock timing includes preprocessing, allocations, layout conversions, actual host/device transfers, resize, grayscale, blur, and host output conversion. CUDA synchronizes before the start and before stopping the timer. Transfers use pageable host memory and blocking copies; no pinned-memory or transfer overlap is assumed, including on Jetson shared DRAM. The boundary is an already-loaded input batch to a host output batch; disk I/O and decoding are excluded.

Stage profiling uses wall-clock timers with CUDA fences around every stage, separately from end-to-end runs. GPU preprocessing is split into host wrapping and device layout/float conversion; these are not individually equivalent to CPU preprocessing. Compare resize, grayscale, and blur stages directly; use full end-to-end timing to judge the complete pipeline. Stage medians need not sum to the end-to-end median. Throughput is total frames divided by total measured end-to-end time, not inverse per-frame median. The speedup is CPU batch median divided by GPU batch median, and is only populated when both succeed and input checksums match.

Memory reports include process peak RSS (Linux `/proc/self/status` `VmHWM`, covering imports, setup, warmup and both timing passes), RSS after warmup, and CUDA allocator peak allocated/reserved bytes since warmup. CPU peak RSS is an absolute process footprint, not incremental algorithm memory; it includes the input workload. GPU worker RSS includes Torch and driver overhead. CUDA allocator peaks exclude driver allocations and external users. Jetson shares physical memory, so RSS and CUDA figures must not be added as independent RAM pools. Peaks are captured before correctness evaluation to avoid counting the extra CPU reference computation. GPU out-of-memory and other execution errors are recorded per case and produce a nonzero exit; no failed case receives a speedup.

The JSON records hardware, CPU affinity, RAM, Jetson/L4T identity where readable, Python/OpenCV/NumPy/PyTorch/CUDA/cuDNN versions, OpenCV build details, source hashes, Git revision/status, configuration, input hashes, raw samples, memory, and final-output MAE, RMSE, maximum absolute difference, PSNR (unit range), exact equality and tolerance status. Perfect-match PSNR is null with `identical=true`, avoiding nonstandard JSON infinity. Similarity evaluation occurs outside timings. These are numerical fidelity measurements of matched algorithms, not detection accuracy claims. Algorithm comparisons require a separate labeled experiment.

## Historical baseline environment diagnosis and observed results

See [`results/baseline.json`](../results/baseline.json) for the complete actual run and [`results/observations.md`](../results/observations.md) for interpretation. No GPU timings or speedups have been fabricated. CUDA numerical parity was unverified in that baseline environment; later reports and the refinement checks use the existing working JetPack environment.

The original baseline aarch64 Tegra environment reports PyTorch `2.15.0+cu132` built with CUDA 13.2. CUDA initialization reports an old driver (version 12020), and NVIDIA memory-manager/NVML initialization also fails. These indicate a runtime/driver compatibility problem and possibly inaccessible device resources; they do not establish that replacing a single component would fix the environment. Drivers and dependencies were left unchanged. The environment diagnostics are read-only.

Interpret future working-CUDA runs by inspecting common stage speedups and end-to-end speedups together. Resize, grayscale and Gaussian blur are GPU candidates, particularly at larger batches/resolutions; that is a hypothesis until measured here. Host input wrapping and host output formatting remain CPU work. If compute stages improve but end-to-end speedup is at or below 1, transfer, conversion, allocation and launch/synchronization overhead eliminate the pipeline benefit. Transfer stage times help quantify that cost; they do not by themselves prove causation. Small workloads often have less compute available to amortize overhead. That baseline makes no measured claim of GPU acceleration; later working-CUDA measurements are linked above.

Implementation references: [PyTorch interpolate](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html), [CUDA synchronization](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html), and [OpenCV filtering and borders](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html).
