"""Input: uint8 NHWC RGB. Output: float32 NHW grayscale in [0, 1]."""
from dataclasses import dataclass
from time import perf_counter
import cv2
import numpy as np


@dataclass(frozen=True)
class Config:
    height: int
    width: int
    kernel: int = 5
    sigma: float = 1.2

    def __post_init__(self):
        if min(self.height, self.width) <= self.kernel // 2:
            raise ValueError('Output dimensions must exceed the reflection padding')
        if self.kernel < 1 or self.kernel % 2 == 0 or not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError('Kernel must be positive and odd; sigma must be finite and positive')


def validate(frames):
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3 or min(frames.shape) < 1:
        raise ValueError('Expected nonempty uint8 NHWC RGB frames')


class CPUBackend:
    def __init__(self, config):
        self.config = config
        self.kernel = cv2.getGaussianKernel(config.kernel, config.sigma, cv2.CV_32F).ravel()

    def sync(self):
        pass

    def run(self, frames, profile=False):
        validate(frames)
        c = self.config
        times = {}
        def stage(name, fn):
            start = perf_counter()
            value = fn()
            if profile:
                times[name] = (perf_counter() - start) * 1000
            return value
        x = stage('preprocess', lambda: frames.astype(np.float32) / np.float32(255))
        x = stage('resize', lambda: np.stack([cv2.resize(f, (c.width, c.height), interpolation=cv2.INTER_LINEAR) for f in x]))
        x = stage('grayscale', lambda: x[..., 0] * np.float32(.299) + x[..., 1] * np.float32(.587) + x[..., 2] * np.float32(.114))
        x = stage('gaussian_blur', lambda: np.stack([cv2.sepFilter2D(f, -1, self.kernel, self.kernel, borderType=cv2.BORDER_REFLECT_101) for f in x]))
        x = stage('output', lambda: np.ascontiguousarray(x))
        return x, times


class GPUBackend:
    def __init__(self, config):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; GPU execution cannot be benchmarked')
        self.torch = torch
        self.config = config
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        k = cv2.getGaussianKernel(config.kernel, config.sigma, cv2.CV_32F).ravel()
        self.kx = torch.as_tensor(k.copy(), device='cuda').reshape(1, 1, 1, -1)
        self.ky = torch.as_tensor(k.copy(), device='cuda').reshape(1, 1, -1, 1)
        self.sync()

    def sync(self):
        self.torch.cuda.synchronize()

    def run(self, frames, profile=False):
        validate(frames)
        torch = self.torch
        F = torch.nn.functional
        c = self.config
        times = {}
        def stage(name, fn):
            if profile:
                self.sync()
            start = perf_counter()
            value = fn()
            if profile:
                self.sync()
                times[name] = (perf_counter() - start) * 1000
            return value
        with torch.inference_mode():
            x = stage('preprocess', lambda: torch.from_numpy(np.ascontiguousarray(frames)))
            x = stage('host_to_device', lambda: x.to('cuda', non_blocking=False))
            x = stage('device_preprocess', lambda: x.permute(0, 3, 1, 2).contiguous().float().div_(255))
            x = stage('resize', lambda: F.interpolate(x, size=(c.height, c.width), mode='bilinear', align_corners=False, antialias=False))
            x = stage('grayscale', lambda: x[:, 0:1] * .299 + x[:, 1:2] * .587 + x[:, 2:3] * .114)
            p = c.kernel // 2
            def blur():
                y = F.conv2d(F.pad(x, (p, p, 0, 0), mode='reflect'), self.kx)
                return F.conv2d(F.pad(y, (0, 0, p, p), mode='reflect'), self.ky)
            x = stage('gaussian_blur', blur)
            x = stage('device_to_host', lambda: x[:, 0].cpu())
            x = stage('output', lambda: np.ascontiguousarray(x.numpy()))
        return x, times
