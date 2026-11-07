"""Stage 1: video ingestion.

Frames are timestamped with the decoder's presentation timestamp (PTS), never index / header fps:
phone footage is often variable-frame-rate (Cabinet-1.MOV declares 120 fps but averages 30 fps,
with 25-42 ms frame gaps). Sampling is by time, so the requested rate holds on VFR video.
"""
from dataclasses import dataclass, field
import json
import shutil
import subprocess

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    opencv_reported_fps: float    # CAP_PROP_FPS: OpenCV 4.8 returns the declared rate (120 for
                                  # Cabinet-1.MOV), OpenCV 5.0 the average (30) - not reliable
    frame_count: int = 0          # frames actually decoded
    duration_s: float = 0.0       # last PTS + mean frame interval
    measured_fps: float = 0.0     # frames decoded / duration: the real average rate
    container_declared_fps: float = None   # ffprobe r_frame_rate
    container_average_fps: float = None    # ffprobe avg_frame_rate

    def as_dict(self):
        return dict(self.__dict__)


@dataclass
class Frame:
    index: int        # decode index in the full video
    t: float          # presentation timestamp, seconds
    image: np.ndarray = field(repr=False)  # full-resolution BGR


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f'cannot open video {path}')
    info = VideoInfo(str(path), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), float(cap.get(cv2.CAP_PROP_FPS) or 0))
    info.container_declared_fps, info.container_average_fps = container_rates(path)
    return cap, info


def iter_frames(path, sample_fps=None, info=None):
    """Yield sampled Frames in decode order. If `info` is given it is completed after the last frame."""
    cap, opened = open_video(path)
    if info is not None:
        info.__dict__.update(opened.__dict__)
    index, next_t, t = 0, 0.0, 0.0
    try:
        while cap.grab():
            # The FFmpeg backend reports a small negative PTS for the first frame of some files.
            t = max(0.0, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            if sample_fps is None or t >= next_t - 1e-6:
                ok, image = cap.retrieve()
                if ok:
                    yield Frame(index, t, image)
                if sample_fps:
                    next_t += 1 / sample_fps
                    if next_t < t:  # skip ahead after long gaps instead of bursting
                        next_t = t + 1 / sample_fps
            index += 1
    finally:
        cap.release()
    if info is not None and index:
        info.frame_count = index
        info.duration_s = t + t / max(index - 1, 1)
        info.measured_fps = index / info.duration_s if info.duration_s else 0.0


def _rate(text):
    num, _, den = text.partition('/')
    try:
        return float(num) / float(den or 1) if float(den or 1) else None
    except ValueError:
        return None


def container_rates(path):
    """(declared, average) frame rate from the container via ffprobe, or (None, None)."""
    if not shutil.which('ffprobe'):
        return None, None
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'stream=r_frame_rate,avg_frame_rate', '-of', 'json', str(path)],
                         capture_output=True, text=True)
    try:
        st = json.loads(out.stdout)['streams'][0]
    except (ValueError, KeyError, IndexError):
        return None, None
    return _rate(st.get('r_frame_rate', '')), _rate(st.get('avg_frame_rate', ''))


def ffprobe_timestamps(path):
    """Container PTS for every video frame via ffprobe (used to verify decoder timestamps), or None."""
    if not shutil.which('ffprobe'):
        return None
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'frame=best_effort_timestamp_time', '-of', 'json', str(path)],
                         capture_output=True, text=True, check=True).stdout
    return [float(f['best_effort_timestamp_time']) for f in json.loads(out)['frames']
            if 'best_effort_timestamp_time' in f]


def resize_to_width(image, width):
    scale = width / image.shape[1]
    if abs(scale - 1) < 1e-9:
        return image, 1.0
    return cv2.resize(image, (width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA), scale
