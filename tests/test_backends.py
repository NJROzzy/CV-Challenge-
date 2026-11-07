import cv2
import numpy as np
import pytest
from cvbench.backends import Config, CPUBackend, GPUBackend
from cvbench.benchmark import similarity, summary, workload


def test_cpu_rgb_contract_and_batch():
    frames = np.zeros((3, 12, 16, 3), np.uint8)
    for i in range(3):
        frames[i, ..., i] = 255
    output, timings = CPUBackend(Config(6, 8)).run(frames, profile=True)
    assert output.shape == (3, 6, 8)
    assert output.dtype == np.float32 and output.flags.c_contiguous
    np.testing.assert_allclose(output[:, 2, 2], [.299, .587, .114], atol=1e-6)
    assert set(timings) == {'preprocess', 'resize', 'grayscale', 'gaussian_blur', 'output'}
    assert all(v >= 0 for v in timings.values())


def test_gaussian_border_impulse():
    frames = np.zeros((1, 7, 9, 3), np.uint8)
    frames[0, 0, 0, :] = 255
    actual, _ = CPUBackend(Config(7, 9, 3, 1)).run(frames)
    kernel = np.exp(-np.arange(-1, 2, dtype=np.float64) ** 2 / 2)
    kernel /= kernel.sum()
    image = frames[0, ..., 0] / 255
    padded = np.pad(image, 1, mode='reflect')
    expected = np.array([[np.sum(padded[y:y+3, x:x+3] * np.outer(kernel, kernel)) for x in range(9)] for y in range(7)])
    np.testing.assert_allclose(actual[0], expected, atol=1e-7)


def test_validation_and_repeatability():
    with pytest.raises(ValueError):
        Config(10, 10, 4)
    with pytest.raises(ValueError):
        CPUBackend(Config(8, 8)).run(np.zeros((1, 8, 8, 3), np.float32))
    assert np.array_equal(workload(12, 16, 4, 7), workload(12, 16, 4, 7))
    assert summary([1, 2, 3])['median_ms'] == 2
    assert similarity(np.zeros(4), np.ones(4))['max_abs'] == 1
    assert not similarity(np.zeros(4), np.ones(4))['within_tolerance']


@pytest.mark.parametrize('shape,target', [((1, 17, 23, 3), (9, 12)), ((4, 24, 32, 3), (12, 16)), ((8, 12, 16, 3), (6, 8))])
@pytest.mark.parametrize('pattern', ['random', 'edge', 'constant'])
def test_cuda_matches_cpu(shape, target, pattern):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable; GPU numerical parity requires real CUDA execution')
    frames = workload(shape[1], shape[2], shape[0], 19)
    if pattern == 'edge':
        frames[:] = 0
        frames[:, 0, :, :] = 255
        frames[:, :, 0, :] = 255
    elif pattern == 'constant':
        frames[:] = [255, 0, 0]
    config = Config(*target)
    expected, _ = CPUBackend(config).run(frames)
    actual, timings = GPUBackend(config).run(frames, profile=True)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert {'host_to_device', 'device_to_host', 'resize', 'grayscale', 'gaussian_blur'} <= timings.keys()
    assert similarity(expected, actual)['within_tolerance']
