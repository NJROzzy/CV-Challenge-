"""End-to-end pipeline: video -> measurements -> annotated images + results.json (+ browser proxy video).

Per-stage wall-clock timings are recorded for every run (decode, detect, registration prep,
frame cache, events/endpoints/registration, annotation, proxy encode).
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import cv2
import numpy as np

from .annotate import render, render_scene
from .detect import ClassicalTapeDetector
from .events import EventBuilder
from .labels import label_measurements
from .register import KeyframeMatcher, Registrar
from .scene import SceneBuilder
from .video import VideoInfo, iter_frames


def make_detector(name):
    if name == 'classical':
        return ClassicalTapeDetector()
    if name == 'hand-assisted':
        from .hands import HandAssistedDetector
        return HandAssistedDetector()
    raise ValueError(f'unknown detector {name!r}')


class StageTimer:
    def __init__(self):
        self.ms = defaultdict(float)
        self.calls = defaultdict(int)

    def add(self, stage, seconds):
        self.ms[stage] += seconds * 1000
        self.calls[stage] += 1

    def as_dict(self):
        return {k: {'total_ms': round(v, 1), 'calls': self.calls[k],
                    'mean_ms': round(v / self.calls[k], 2)} for k, v in self.ms.items()}


def make_proxy(video, out_path, height=960):
    """Browser-playable H.264 copy that keeps the original (variable) timestamps.
    (-vsync passthrough: ffmpeg 4.4 on JetPack predates the newer -fps_mode spelling.)"""
    if not shutil.which('ffmpeg'):
        return None
    cmd = ['ffmpeg', '-v', 'error', '-y', '-i', str(video), '-an', '-vf', f'scale=-2:{height}',
           '-vsync', 'passthrough', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '26',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out_path)]
    subprocess.run(cmd, check=True)
    return out_path


def run(video, out_dir, sample_fps=10.0, detector='classical', proxy=True, log=print):
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = StageTimer()
    wall0 = time.perf_counter()
    det = make_detector(detector)
    reg = Registrar()
    info = VideoInfo(str(video), 0, 0, 0.0)
    observations, sharpness, cache = [], {}, {}

    frames = iter_frames(video, sample_fps, info)
    while True:
        t0 = time.perf_counter()
        try:
            f = next(frames)
        except StopIteration:
            break
        timer.add('decode_and_sample', time.perf_counter() - t0)

        t0 = time.perf_counter()
        obs = det.detect(f.image, f.index, f.t)
        timer.add('detect', time.perf_counter() - t0)

        t0 = time.perf_counter()
        reg.add(f.index, f.image, obs)
        gray = reg.frames[f.index][0]
        sharpness[f.index] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        timer.add('registration_prep', time.perf_counter() - t0)

        t0 = time.perf_counter()
        cache[f.index] = (f.t, cv2.imencode('.jpg', f.image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1])
        timer.add('frame_cache', time.perf_counter() - t0)
        if obs is not None:
            observations.append(obs)
    samples = len(cache)
    log(f'{video.name}: {info.frame_count} frames, {info.duration_s:.1f}s; {samples} samples, '
        f'tape detected in {len(observations)}')

    t0 = time.perf_counter()
    builder = EventBuilder(reg, sharpness, (info.width, info.height))
    measurements = builder.build(observations)
    label_measurements(measurements, reg)
    timer.add('events_endpoints_registration', time.perf_counter() - t0)

    t0 = time.perf_counter()
    matcher = KeyframeMatcher(lambda idx: cv2.imdecode(cache[idx][1], cv2.IMREAD_COLOR) if idx in cache else None,
                              {k: v[1] for k, v in reg.frames.items()})
    scene_builder = SceneBuilder(reg, sharpness, reg.clutter, {k: v[0] for k, v in cache.items()},
                                 matcher=matcher, tape_frames={o.frame_index for o in observations})
    scenes = scene_builder.build(measurements)
    timer.add('scene_registration', time.perf_counter() - t0)

    t0 = time.perf_counter()
    for pattern in ('measurement_*.jpg', 'scene_*.jpg', 'diagnostics/candidate_*.jpg'):
        for old in out_dir.glob(pattern):           # never leave images from a previous run
            old.unlink()
    (out_dir / 'diagnostics').mkdir(exist_ok=True)
    decode = lambda idx: cv2.imdecode(cache[idx][1], cv2.IMREAD_COLOR)
    validated, candidates = [], []
    for m in measurements:
        md = m.as_dict()
        if m.context_frame is not None:
            md['context_t'] = round(float(cache[m.context_frame][0]), 3)
        ctx_img = H_ctx = None
        if m.context_frame is not None:
            r = reg.register(m.context_frame, m.canvas_frame)
            if r.ok:
                ctx_img, H_ctx = decode(m.context_frame), r.H
                md['context_registration'] = r.as_dict()
                reg_xy = getattr(m.hook, 'registered_xy', None)
                if reg_xy is not None and m.hook.status == 'registered':
                    # Apply the hook's straight-blade correction to the context image as well, so
                    # the mosaic agrees with the plotted hook (recorded, shown in the caption).
                    dx, dy = m.hook.x - reg_xy[0], m.hook.y - reg_xy[1]
                    H_ctx = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64) @ H_ctx
                    md['context_shift_px'] = round(float(np.hypot(dx, dy)), 1)
        img = render(md, decode(m.canvas_frame), ctx_img, H_ctx, video.name)
        if m.status == 'validated':
            name = f'measurement_{m.id:02d}.jpg'
            validated.append(md)
        else:
            name = f'diagnostics/candidate_{m.id:02d}.jpg'
            candidates.append(md)
        cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        md['image'] = name
    by_id = {md['id']: md for md in validated}
    scene_dicts = []
    for i, sc in enumerate(scenes, 1):
        sd = sc.as_dict()
        if sc.kind == 'reading_frame' and len(sc.placements) == 1:
            # No other view registered: the measurement's own (possibly mosaic) image is the scene.
            sd['image'] = by_id[sc.placements[0].measurement_id]['image']
        else:
            sd['image'] = f'scene_{i:02d}.jpg'
            cv2.imwrite(str(out_dir / sd['image']), render_scene(sd, decode(sc.frame), by_id, video.name),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        scene_dicts.append(sd)
    timer.add('annotate', time.perf_counter() - t0)

    proxy_name = None
    if proxy:
        t0 = time.perf_counter()
        try:
            existing = out_dir / 'video.mp4'
            if existing.exists() and existing.stat().st_mtime > video.stat().st_mtime:
                proxy_name = 'video.mp4'       # browser copy of this exact source already made
            elif make_proxy(video, existing):
                proxy_name = 'video.mp4'
        except subprocess.CalledProcessError as exc:
            log(f'proxy video failed: {exc}')
        timer.add('viewer_proxy_encode', time.perf_counter() - t0)

    results = {
        'schema': 'reekon.results/2',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'video': info.as_dict() | {'name': video.name},
        'config': {'sample_fps': sample_fps, 'detector': detector, 'python': platform.python_version(),
                   'opencv': cv2.__version__, 'numpy': np.__version__},
        'pipeline_seconds': round(time.perf_counter() - wall0, 2),
        'pipeline_seconds_excluding_proxy': round(time.perf_counter() - wall0 - timer.ms.get('viewer_proxy_encode', 0) / 1000, 2),
        'stage_timings': timer.as_dict(),
        'counts': {'samples': samples, 'tape_observations': len(observations),
                   'candidate_events': len(measurements), 'measurements': len(validated),
                   'uncertain': sum(c['status'] == 'uncertain' for c in candidates),
                   'rejected': sum(c['status'] == 'rejected' for c in candidates),
                   'scene_images': len(scene_dicts),
                   'registered_view_scenes': sum(sd['kind'] == 'registered_view' for sd in scene_dicts),
                   'clean_view_scenes': sum(sd['kind'] == 'clean_view' for sd in scene_dicts)},
        'identity': builder.identity,
        'episodes': builder.log[0]['episodes'] if builder.log else [],
        'measurements': validated,
        'candidates': candidates,
        'scenes': scene_dicts,
        'scene_log': scene_builder.log,
        'observations': [o.as_dict() for o in observations],
        'sampled_times': [round(cache[k][0], 4) for k in sorted(cache)],
        'proxy_video': proxy_name,
    }
    (out_dir / 'results.json').write_text(json.dumps(results, indent=1, default=_json_default))
    log(f'  {len(measurements)} candidate event(s): {len(validated)} validated, '
        f'{len(candidates)} uncertain/rejected; {len(scene_dicts)} scene image(s); outputs in {out_dir}')
    return results


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
