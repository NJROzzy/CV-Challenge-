"""Failure contracts at pipeline boundaries; no footage or optional GPU required."""
import numpy as np
import pytest

from reekon import pipeline, video


@pytest.mark.parametrize('rate', [0, -1, float('nan'), float('inf')])
def test_invalid_rate_fails_before_opening_video_or_creating_output(tmp_path, rate):
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='finite and positive'):
        pipeline.run(tmp_path / 'absent.mp4', out, sample_fps=rate)
    assert not out.exists()
    with pytest.raises(ValueError, match='finite and positive'):
        next(video.iter_frames(tmp_path / 'absent.mp4', sample_fps=rate))


def test_empty_decode_does_not_publish_results(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, 'iter_frames', lambda *args: iter(()))
    with pytest.raises(ValueError, match='no frames decoded'):
        pipeline.run('empty.mp4', tmp_path, proxy=False)
    assert not (tmp_path / 'results.json').exists()


def test_image_write_failure_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.cv2, 'imwrite', lambda *args: False)
    with pytest.raises(OSError, match='cannot write image'):
        pipeline._write_image(tmp_path / 'measurement.jpg', np.zeros((8, 8, 3), np.uint8))
