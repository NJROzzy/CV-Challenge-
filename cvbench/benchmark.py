"""Each backend/workload runs in a fresh process to isolate RAM high-water marks."""
import argparse
from contextlib import redirect_stderr
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import warnings
import cv2
import numpy as np
import psutil
from .backends import CPUBackend, GPUBackend, Config

RESOLUTIONS = {'480p': (480, 640), '720p': (720, 1280), '1080p': (1080, 1920)}


def summary(values):
    return {'median_ms': float(np.median(values)), 'p95_ms': float(np.percentile(values, 95)), 'samples_ms': values}


def workload(height, width, batch, seed):
    # The same seed/shape regenerates byte-identical input in independent workers.
    return np.random.default_rng(seed).integers(0, 256, (batch, height, width, 3), dtype=np.uint8)


def similarity(reference, actual, atol=2e-5):
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(delta ** 2))
    return {'mae': float(np.mean(np.abs(delta))), 'rmse': mse ** .5,
            'max_abs': float(np.max(np.abs(delta))),
            'psnr_db': float(10 * np.log10(1 / mse)) if mse else None,
            'identical': bool(np.array_equal(reference, actual)),
            'within_tolerance': bool(np.allclose(reference, actual, atol=atol, rtol=0)), 'atol': atol}


def read_optional(path):
    try:
        return Path(path).read_text().strip('\x00\n')
    except (OSError, TypeError, UnicodeError):
        # Some Jetson sysfs nodes fail reads even when directory entries exist.
        return None


def peak_rss_bytes():
    # /proc tracks this process address space; getrusage can retain a parent's
    # pre-exec high-water mark in subprocesses on Linux.
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmHWM:'):
            return int(line.split()[1]) * 1024
    return None


def environment():
    import torch
    with warnings.catch_warnings(record=True) as captured, redirect_stderr(io.StringIO()) as stderr:
        warnings.simplefilter('always')
        available = torch.cuda.is_available()
        count = torch.cuda.device_count()
    return {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'platform': platform.platform(), 'machine': platform.machine(),
            'jetson_model': read_optional('/proc/device-tree/model'),
            'l4t_release': read_optional('/etc/nv_tegra_release'),
            'cpu': platform.processor(), 'cpu_count': os.cpu_count(),
            'cpu_affinity': psutil.Process().cpu_affinity(),
            'ram_total_bytes': psutil.virtual_memory().total,
            'python': sys.version, 'numpy': np.__version__, 'opencv': cv2.__version__,
            'opencv_build': cv2.getBuildInformation(), 'torch': torch.__version__,
            'torch_cuda_build': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'cuda_available': available, 'cuda_device_count': count,
            'cuda_devices': [str(torch.cuda.get_device_properties(i)) for i in range(count)] if available else [],
            'cuda_warnings': [str(w.message) for w in captured], 'cuda_stderr': stderr.getvalue(),
            'power_mode': read_optional('/etc/nvpmodel.conf'),
            'thermal_millicelsius': {str(p): read_optional(p) for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')},
            'cpu_frequency_khz': {str(p): read_optional(p) for p in Path('/sys/devices/system/cpu').glob('cpu*/cpufreq/scaling_cur_freq')},
            'git_revision': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() or None,
            'git_status': subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout,
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}


def worker(c):
    cv2.setNumThreads(c['threads'])
    cv2.ocl.setUseOpenCL(False)
    frames = workload(c['height'], c['width'], c['batch'], c['seed'])
    cfg = Config(c['height'] // 2, c['width'] // 2, c['kernel'], c['sigma'])
    backend = CPUBackend(cfg) if c['backend'] == 'cpu' else GPUBackend(cfg)
    for _ in range(c['warmup']):
        backend.run(frames)
        backend.sync()
    rss_baseline = psutil.Process().memory_info().rss
    if c['backend'] == 'gpu':
        backend.torch.cuda.reset_peak_memory_stats()
    e2e = []
    for _ in range(c['repetitions']):
        backend.sync()
        start = perf_counter()
        output, _ = backend.run(frames)
        backend.sync()  # Always included before stopping CUDA end-to-end timer.
        e2e.append((perf_counter() - start) * 1000)
        del output
    stages = {}
    # Separate synchronized profiling pass: stage fences do not distort E2E.
    for _ in range(c['repetitions']):
        output, timings = backend.run(frames, profile=True)
        for name, value in timings.items():
            stages.setdefault(name, []).append(value)
        del output
    result = {'status': 'ok', 'backend': c['backend'], 'input_sha256': hashlib.sha256(frames.tobytes()).hexdigest(),
              'end_to_end': summary(e2e), 'stages': {k: summary(v) for k, v in stages.items()},
              'throughput_fps': c['batch'] * len(e2e) * 1000 / sum(e2e),
              'memory': {'rss_after_warmup_bytes': rss_baseline,
                         'process_peak_rss_bytes': peak_rss_bytes(),
                         'cuda_peak_allocated_bytes': None, 'cuda_peak_reserved_bytes': None}}
    if c['backend'] == 'gpu':
        result['memory'].update(cuda_peak_allocated_bytes=backend.torch.cuda.max_memory_allocated(),
                                cuda_peak_reserved_bytes=backend.torch.cuda.max_memory_reserved())
        actual, _ = backend.run(frames)
        reference, _ = CPUBackend(cfg).run(frames)
        result['similarity'] = similarity(reference, actual)
    else:
        result['similarity'] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['cpu', 'gpu', 'both'], default='both')
    parser.add_argument('--resolutions', nargs='+', choices=list(RESOLUTIONS), default=list(RESOLUTIONS))
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 4, 8])
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repetitions', type=int, default=30)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--kernel', type=int, default=5)
    parser.add_argument('--sigma', type=float, default=1.2)
    parser.add_argument('--output', default='results/benchmark.json')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        c = json.loads(args.worker)
        try:
            print(json.dumps(worker(c), allow_nan=False))
        except Exception as exc:
            print(json.dumps({'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}))
        return
    if min(args.batches + [args.warmup, args.repetitions, args.threads]) < 1:
        parser.error('Batch sizes, warmup, repetitions, and threads must be positive')
    Config(240, 320, args.kernel, args.sigma)
    report = {'schema_version': 1, 'environment': environment(), 'configuration': vars(args),
              'workload': {'type': 'seeded uniform RGB noise', 'input': 'uint8 NHWC RGB',
                           'resize': 'half input width and height; bilinear half-pixel; no antialias',
                           'output': 'float32 NHW grayscale [0,1]', 'border': 'reflect101',
                           'grayscale_coefficients': [.299, .587, .114]}, 'cases': []}
    for resolution in args.resolutions:
        height, width = RESOLUTIONS[resolution]
        for batch in args.batches:
            case = {'resolution': resolution, 'input_shape': [batch, height, width, 3], 'backends': {}}
            for backend in (['cpu', 'gpu'] if args.backend == 'both' else [args.backend]):
                if backend == 'gpu' and not report['environment']['cuda_available']:
                    result = {'status': 'unavailable', 'reason': 'CUDA initialization unavailable; see environment diagnostics'}
                else:
                    c = dict(height=height, width=width, batch=batch, seed=args.seed, backend=backend,
                             kernel=args.kernel, sigma=args.sigma, warmup=args.warmup,
                             repetitions=args.repetitions, threads=args.threads)
                    proc = subprocess.run([sys.executable, '-m', 'cvbench', '--worker', json.dumps(c)], capture_output=True, text=True)
                    try:
                        result = json.loads(proc.stdout)
                    except ValueError:
                        result = {'status': 'error', 'returncode': proc.returncode, 'stderr': proc.stderr, 'stdout': proc.stdout}
                case['backends'][backend] = result
            cpu, gpu = case['backends'].get('cpu', {}), case['backends'].get('gpu', {})
            case['comparison'] = None
            if cpu.get('status') == gpu.get('status') == 'ok':
                matched = cpu['input_sha256'] == gpu['input_sha256']
                case['comparison'] = {'inputs_identical': matched,
                    'speedup_cpu_over_gpu': cpu['end_to_end']['median_ms'] / gpu['end_to_end']['median_ms'] if matched else None,
                    'stage_speedup_cpu_over_gpu': {k: cpu['stages'][k]['median_ms'] / gpu['stages'][k]['median_ms'] for k in ('resize', 'grayscale', 'gaussian_blur')} if matched else None}
            report['cases'].append(case)
            print(f'{resolution} batch={batch}: ' + ', '.join(f'{k}={v["status"]}' for k, v in case['backends'].items()), flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Report: {path}')
    if any(r['status'] == 'error' for c in report['cases'] for r in c['backends'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
