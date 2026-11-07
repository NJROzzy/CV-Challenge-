"""Tests for the measurement pipeline on synthetic inputs with known answers.

The real-footage regression test (Cabinet-1.MOV) is opt-in because it takes ~70 s:
    REEKON_FOOTAGE_TESTS=1 .venv/bin/python -m pytest tests/test_reekon.py -k footage
"""
from dataclasses import dataclass
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading

import cv2
import numpy as np
import pytest

from reekon import video as V
from reekon.annotate import render
from reekon.detect import ClassicalTapeDetector, Endpoint, TapeObservation
from reekon.events import EventBuilder, housing_phases, mode_cluster, split_episodes
from reekon.register import Registrar, map_points

ROOT = Path(__file__).resolve().parent.parent
FOOTAGE = ROOT / 'data' / 'inbox' / 'Cabinet-1.MOV'


# ---------------------------------------------------------------------------- synthetic scenes
def textured_background(h, w, seed=0):
    """Light, cabinet-like background with enough corners for tracking."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 225, np.uint8)
    for _ in range(140):
        x, y = rng.integers(0, w), rng.integers(0, h)
        s = rng.integers(4, 14)
        shade = int(rng.integers(150, 205))
        cv2.rectangle(img, (int(x), int(y)), (int(x + s), int(y + s)), (shade, shade, shade), -1)
    return img


def draw_tape(img, hook, housing_end, width=12, housing=True):
    """Black blade with white ticks from `hook` to `housing_end`, plus a wide dark housing beyond it."""
    hook, housing_end = np.float64(hook), np.float64(housing_end)
    d = (housing_end - hook) / np.linalg.norm(housing_end - hook)
    n = np.array([-d[1], d[0]])
    cv2.line(img, tuple(hook.astype(int)), tuple(housing_end.astype(int)), (20, 20, 20), width)
    length = np.linalg.norm(housing_end - hook)
    for k in np.arange(8, length - 4, 9):
        p = hook + d * k
        a, b = p - n * width * 0.3, p + n * width * 0.3
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (240, 240, 240), 2)
    if housing:
        c = housing_end + d * 4 * width
        box = cv2.boxPoints(((float(c[0]), float(c[1])), (8 * width, 5 * width),
                             float(np.degrees(np.arctan2(d[1], d[0])))))
        cv2.fillPoly(img, [box.astype(np.int32)], (25, 25, 25))
        cv2.circle(img, tuple(c.astype(int)), width, (230, 230, 230), -1)  # bright LCD inside
    return img


# ---------------------------------------------------------------------------- video ingestion
def write_video(path, frames, fps=30):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


def test_iter_frames_samples_by_time(tmp_path):
    frames = [np.full((64, 96, 3), i * 4 % 255, np.uint8) for i in range(60)]
    path = tmp_path / 'clip.mp4'
    write_video(path, frames, fps=30)
    info = V.VideoInfo(str(path), 0, 0, 0.0)
    got = list(V.iter_frames(path, sample_fps=5, info=info))
    ts = [f.t for f in got]
    assert info.frame_count == 60 and abs(info.duration_s - 2.0) < 0.05
    assert all(b > a for a, b in zip(ts, ts[1:]))
    assert 9 <= len(got) <= 11                                  # 2 s at 5 samples/s
    assert np.allclose(np.diff(ts), 0.2, atol=0.034)            # spaced by time, within a frame
    if shutil.which('ffprobe'):
        pts = V.ffprobe_timestamps(path)
        assert len(pts) == 60
        decoded = [f.t for f in V.iter_frames(path)]
        assert np.allclose(decoded[1:], pts[1:len(decoded)], atol=1e-3)


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='ffmpeg needed to build a VFR clip')
def test_variable_frame_rate_uses_presentation_timestamps(tmp_path):
    """A clip whose frame gaps vary must be timestamped by PTS, not index / fps."""
    src = tmp_path / 'cfr.mp4'
    write_video(src, [np.full((48, 64, 3), i * 9 % 255, np.uint8) for i in range(30)], fps=30)
    vfr = tmp_path / 'vfr.mp4'
    # Duplicate-free VFR: timestamps follow t = 0.05 * n^1.2 (gaps grow over time).
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(src), '-vf', "setpts='0.05*pow(N,1.2)/TB'",
                    '-vsync', 'vfr', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(vfr)], check=True)
    ts = np.array([f.t for f in V.iter_frames(vfr)])
    expected = 0.05 * np.arange(len(ts)) ** 1.2
    assert len(ts) == 30
    assert np.allclose(ts, expected, atol=0.02)
    assert np.ptp(np.diff(ts)) > 0.01                           # really variable


# ---------------------------------------------------------------------------- detection
def test_detects_hook_and_housing_ends():
    img = draw_tape(textured_background(640, 480), hook=(120, 90), housing_end=(300, 470))
    obs = ClassicalTapeDetector().detect(img)
    assert obs is not None and obs.housing_detected
    assert obs.hook.visible and obs.housing.visible
    assert np.linalg.norm(obs.hook.xy - (120, 90)) < 15
    assert np.linalg.norm(obs.housing.xy - (300, 470)) < 20      # where the blade enters the housing
    assert obs.evidence['tick_texture'] > 0.06


def test_end_cut_off_by_border_is_not_reported_as_seen():
    # Blade enters from the left border; hook in view, no housing in frame.
    img = draw_tape(textured_background(640, 480), hook=(300, 320), housing_end=(-60, 340), housing=False)
    obs = ClassicalTapeDetector().detect(img)
    assert obs is not None and not obs.housing_detected
    assert obs.hook.visible and np.linalg.norm(obs.hook.xy - (300, 320)) < 15
    assert not obs.housing.visible                              # never claim an unobserved end


def test_rejects_scenes_without_a_tape():
    det = ClassicalTapeDetector()
    blank = textured_background(640, 480)
    assert det.detect(blank) is None
    gap = blank.copy()
    cv2.line(gap, (240, 0), (240, 639), (30, 30, 30), 2)           # thin cabinet gap: opened away
    assert det.detect(gap) is None
    solid = blank.copy()
    cv2.line(solid, (60, 300), (420, 330), (20, 20, 20), 12)       # dark strip without tick marks
    assert det.detect(solid) is None


# ---------------------------------------------------------------------------- registration
def test_chained_registration_recovers_known_motion():
    base = textured_background(900, 900, seed=3)
    reg = Registrar(work_width=480)
    shifts = [(i * 6.0, i * 3.0) for i in range(8)]               # camera pans 6 px / 3 px per frame
    for i, (dx, dy) in enumerate(shifts):
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        reg.add(i, cv2.warpAffine(base, M, (900, 900), borderMode=cv2.BORDER_REFLECT))
    r = reg.register(0, 7)
    assert r.ok and r.links == 7
    p = map_points(r.H, [[450, 450]])[0]
    assert np.linalg.norm(p - (450 - 42, 450 - 21)) < 2.0          # scene point moves opposite to camera


# ---------------------------------------------------------------------------- events
def _obs(i, t, hook, housing, hook_visible=True, housing_detected=True, angle=90.0, conf=0.9):
    return TapeObservation(i, t, 'test', conf, angle, 10.0,
                           Endpoint(*hook, 'hook', hook_visible), Endpoint(*housing, 'housing', housing_detected),
                           housing_detected)


class IdentityRegistrar:
    """All frames share one viewpoint (static camera)."""
    @dataclass
    class R:
        ok: bool = True
        H: np.ndarray = None
        links: int = 0
        min_inliers: int = 100
        refine_inliers: int = 0

        def as_dict(self):
            return {}

    def register(self, src, dst):
        return self.R(H=np.eye(3), links=0 if src == dst else 1)


def test_mode_cluster_ignores_outliers():
    pts = [[100, 100]] * 6 + [[103, 98], [500, 500], [505, 497], [900, 50]]
    center, members, others = mode_cluster(pts, radius=20)
    assert np.allclose(center, [100, 100], atol=3) and members.sum() == 7
    assert any(n == 2 for _, n in others)


def test_episode_split_and_housing_phases():
    obs = [_obs(i, i * 0.1, (100, 100), (100, 900), angle=90) for i in range(40)]
    obs += [_obs(100 + i, 4.0 + i * 0.1, (100, 500), (800, 500), angle=0) for i in range(20)]
    eps = split_episodes(obs)
    assert [len(e) for e in eps] == [40, 20]                       # direction change splits
    # Housing out of view for ~1 s (camera looks at the hook): two readings, 0-1.1 s and 2.2-3.9 s.
    looked_away = [_obs(o.frame_index, o.t, (100, 100), (100, 900), housing_detected=not 1.15 < o.t < 2.15)
                   for o in eps[0]]
    phases = housing_phases(looked_away)
    assert [(round(p[0].t, 1), round(p[-1].t, 1)) for p in phases] == [(0.0, 1.1), (2.2, 3.9)]
    # Readings shorter than 0.8 s are not measurements.
    assert housing_phases(looked_away[:6]) == []


def test_builder_reports_measurement_and_never_invents_a_hook():
    sharp = {}
    obs = [_obs(i, i * 0.1, (200 + i % 3, 100), (200, 900)) for i in range(30)]
    for o in obs:
        sharp[o.frame_index] = 1.0
    ms = EventBuilder(IdentityRegistrar(), sharp, (480, 1000)).build(obs)
    assert len(ms) == 1
    m = ms[0]
    assert m.hook.status == 'observed' and m.housing.status == 'observed'
    assert abs(m.hook.x - 201) < 3 and abs(m.housing.y - 900) < 1

    no_hook = [_obs(i, i * 0.1, (200, 0), (200, 900), hook_visible=False) for i in range(30)]
    ms = EventBuilder(IdentityRegistrar(), sharp, (480, 1000)).build(no_hook)
    assert len(ms) == 1 and ms[0].hook.status == 'missing' and ms[0].hook.x is None
    assert ms[0].confidence == 0.0


def test_hook_sightings_of_another_placement_are_rejected():
    """Hook seen on a different line (e.g. the neighbouring door) must not be used."""
    sharp = {i: 1.0 for i in range(60)}
    reading = [_obs(i, i * 0.1, (200, 0), (200, 900), hook_visible=False) for i in range(30)]
    elsewhere = [_obs(30 + i, 3.0 + i * 0.05, (700, 100), (700, 950), housing_detected=False) for i in range(10)]
    ms = EventBuilder(IdentityRegistrar(), sharp, (1000, 1000)).build(reading + elsewhere)
    assert len(ms) == 1 and ms[0].hook.status == 'missing'


# ---------------------------------------------------------------------------- annotation
def test_render_adds_caption_band_and_skips_missing_endpoint():
    canvas = np.full((400, 300, 3), 200, np.uint8)
    m = {'id': 1, 'start_t': 1.0, 'end_t': 2.0, 'canvas_t': 1.5,
         'hook': {'status': 'missing', 'x': None, 'y': None, 'confidence': 0, 'support': 0, 'candidates': 4},
         'housing': {'status': 'observed', 'x': 150.0, 'y': 300.0, 'confidence': 0.9, 'support': 9,
                     'candidates': 10, 'spread_px': 4.0}}
    out = render(m, canvas, video_name='x.mov')
    assert out.shape[1] == 300 and out.shape[0] > 400              # caption added above, not over
    band = out.shape[0] - 400
    assert (out[band:band + 150] == 200).all()                     # nothing drawn where a hook would go


# ---------------------------------------------------------------------------- viewer server
def test_server_supports_ranges_and_blocks_other_paths(tmp_path):
    from http.server import ThreadingHTTPServer
    from reekon.serve import Handler
    (tmp_path / 'v').mkdir()
    data = bytes(range(256)) * 4
    (tmp_path / 'v' / 'video.mp4').write_bytes(data)
    Handler.mounts = {'viewer': ROOT / 'viewer', 'outputs': tmp_path, 'bench': ROOT / 'results'}
    Handler.log_message = lambda *a: None
    srv = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        def get(path, headers=None):
            c = HTTPConnection('127.0.0.1', srv.server_address[1])
            c.request('GET', path, headers=headers or {})
            r = c.getresponse()
            return r.status, r.read(), r
        status, body, r = get('/outputs/v/video.mp4', {'Range': 'bytes=10-19'})
        assert status == 206 and body == data[10:20] and r.getheader('Content-Range') == f'bytes 10-19/{len(data)}'
        status, body, _ = get('/outputs/v/video.mp4', {'Range': 'bytes=-5'})
        assert status == 206 and body == data[-5:]
        assert get('/')[0] == 200
        assert get('/reekon/serve.py')[0] == 404
        assert get('/outputs/../reekon/serve.py')[0] == 404
        assert get('/outputs/%2e%2e/reekon/serve.py')[0] == 404
    finally:
        srv.shutdown()


# ---------------------------------------------------------------------------- end to end
def test_pipeline_end_to_end_on_synthetic_video(tmp_path):
    from reekon.pipeline import run
    frames = []
    bg = textured_background(1100, 800, seed=5)
    for i in range(45):                                            # 1.5 s, static camera, steady tape
        f = bg[30:670, 160:640].copy()
        frames.append(draw_tape(f, hook=(140, 80), housing_end=(260, 470)))
    path = tmp_path / 'synthetic.mp4'
    write_video(path, frames, fps=30)
    res = run(path, tmp_path / 'out', sample_fps=10, proxy=False, log=lambda *a: None)
    assert res['counts']['measurements'] == 1
    m = res['measurements'][0]
    assert m['hook']['status'] == 'observed' and m['housing']['status'] == 'observed'
    assert np.hypot(m['hook']['x'] - 140, m['hook']['y'] - 80) < 15
    assert np.hypot(m['housing']['x'] - 260, m['housing']['y'] - 470) < 20
    assert (tmp_path / 'out' / m['image']).is_file()
    saved = json.loads((tmp_path / 'out' / 'results.json').read_text())
    assert {'decode_and_sample', 'detect', 'events_endpoints_registration', 'annotate'} <= saved['stage_timings'].keys()


@pytest.mark.skipif(os.environ.get('REEKON_FOOTAGE_TESTS') != '1' or not FOOTAGE.exists(),
                    reason='opt-in: REEKON_FOOTAGE_TESTS=1 and data/inbox/Cabinet-1.MOV')
def test_footage_cabinet1_regression(tmp_path):
    """Four measurements were verified by eye on Cabinet-1.MOV (two door heights, two widths).
    This guards that the pipeline still finds four, each with both endpoints."""
    from reekon.pipeline import run
    res = run(FOOTAGE, tmp_path, proxy=False, log=lambda *a: None)
    ms = res['measurements']
    assert len(ms) == 4
    assert all(m['hook']['status'] != 'missing' and m['housing']['status'] != 'missing' for m in ms)
    vertical = [m for m in ms if 60 < m['angle_deg'] < 120]
    assert len(vertical) == 2                                      # the two door heights
