"""Stage 4c: scene canvases - put validated measurements onto the least obstructed reliable view.

A measurement's reading frame usually shows the tape, the housing and hands in front of the
measured object. The deliverable is a clear image of the object with the measurement drawn on it,
so for every validated measurement we look for nearby frames that

  * are sharp and little obstructed (tape, housing and skin cover a small part of the frame), and
  * register reliably to the reading frame: both endpoints are carried over with local feature
    motion (no single global homography across surfaces at different depths), carried back through
    independently computed reverse links, and must return close to where they started (cycle
    check) and land inside the frame.

Measurements are grouped greedily onto as few such scene frames as possible. A measurement with
no reliable clear view keeps its own reading frame as its canvas - it is never placed by an
unvalidated transform.
"""
from dataclasses import dataclass, field

import numpy as np

from .config import ROUND_TRIP_TOLERANCE


@dataclass
class Placement:
    measurement_id: int
    hook_xy: list
    housing_xy: list
    cycle_error_px: list
    links: int
    local_moves: int
    global_moves: int
    source: str = 'registered'    # 'reading_frame': this IS the frame the endpoints were localised in


@dataclass
class SceneCanvas:
    frame: int
    t: float
    kind: str                     # 'clean_view', 'registered_view' or 'reading_frame' (fallback)
    clarity: float
    placements: list = field(default_factory=list)

    def as_dict(self):
        return {'frame': self.frame, 't': self.t, 'kind': self.kind, 'clarity': round(self.clarity, 3),
                'placements': [p.__dict__ for p in self.placements]}


class SceneBuilder:
    def __init__(self, registrar, sharpness, clutter, times, window_s=10.0, candidates_per_measurement=30,
                 min_separation_s=0.4, tol_frac=ROUND_TRIP_TOLERANCE, matcher=None, tape_frames=None,
                 clean_max_clutter=0.05, clean_candidates=25):
        self.matcher = matcher            # KeyframeMatcher for direct mapping into clean frames (optional)
        self.tape_frames = set(tape_frames or ())
        self.clean_max_clutter = clean_max_clutter
        self.clean_k = clean_candidates
        self.reg = registrar
        self.sharpness = sharpness      # frame -> variance of Laplacian
        self.clutter = clutter          # frame -> fraction covered by tape/housing/skin
        self.times = times              # frame -> t
        self.window_s = window_s
        self.k = candidates_per_measurement
        self.min_sep = min_separation_s
        self.tol_frac = tol_frac
        self.log = []

    def clarity(self, f, ref_sharp):
        sharp = min(1.0, self.sharpness.get(f, 0.0) / (ref_sharp or 1.0))
        return sharp * (1.0 - self.clutter.get(f, 1.0)) ** 2

    def _candidates(self, m, ref_sharp):
        lo, hi = m.start_t - self.window_s, m.end_t + self.window_s
        frames = [f for f, t in self.times.items() if lo <= t <= hi]
        frames.sort(key=lambda f: -self.clarity(f, ref_sharp))
        chosen = []
        for f in frames:
            if all(abs(self.times[f] - self.times[g]) >= self.min_sep for g in chosen):
                chosen.append(f)
            if len(chosen) == self.k:
                break
        return chosen

    def clean_frames(self, ref_sharp):
        """Frames that show the scene without the act of measuring: no tape detected and almost no
        tape/housing/skin pixels. Sharpest first, at least min_sep apart; taken from the whole video
        (wide views are usually at the start/end, far from the readings)."""
        frames = [f for f in self.times if f not in self.tape_frames
                  and self.clutter.get(f, 1.0) <= self.clean_max_clutter]
        frames.sort(key=lambda f: -self.clarity(f, ref_sharp))
        chosen = []
        for f in frames:
            if all(abs(self.times[f] - self.times[g]) >= self.min_sep for g in chosen):
                chosen.append(f)
            if len(chosen) == self.clean_k:
                break
        return chosen

    def _clean_options(self, valid, ref_sharp):
        options = {}
        if self.matcher is None:
            return options
        clean = self.clean_frames(ref_sharp)
        for m in valid:
            pts = [[m.hook.x, m.hook.y], [m.housing.x, m.housing.y]]
            why, ok_n = {}, 0
            for f in clean:
                mapped, ok, st = self.matcher.map_validated(m.canvas_frame, f, pts)
                if not ok:
                    why[st['reason']] = why.get(st['reason'], 0) + 1
                    continue
                ok_n += 1
                options.setdefault(f, {})[m.id] = Placement(
                    m.id, [round(float(v), 1) for v in mapped[0]], [round(float(v), 1) for v in mapped[1]],
                    st['cycle_error_px'], 1, 0, 0, 'registered')
            self.log.append({'measurement': m.id, 'stage': 'clean_view', 'views_tried': len(clean),
                             'views_ok': ok_n, 'rejected_because': why})
        return options

    def _cover(self, options, unplaced, ref_sharp, kind_if_registered):
        canvases = []
        while unplaced:
            best = max(options, default=None, key=lambda f: (len(unplaced & options[f].keys()),
                                                             self.clarity(f, ref_sharp)))
            if best is None or not unplaced & options[best].keys():
                break
            ids = sorted(unplaced & options[best].keys())
            placements = [options[best][i] for i in ids]
            kind = kind_if_registered if any(pl.source == 'registered' for pl in placements) else 'reading_frame'
            canvases.append(SceneCanvas(best, round(self.times[best], 3), kind,
                                        self.clarity(best, ref_sharp), placements))
            unplaced -= set(ids)
        return canvases

    def build(self, measurements):
        valid = [m for m in measurements if m.status == 'validated']
        if not valid:
            return []
        ref_sharp = float(np.percentile(list(self.sharpness.values()), 90))
        # 1) clean views (no tape, no hands), mapped directly from each reading frame and validated.
        unplaced = {m.id for m in valid}
        canvases = self._cover(self._clean_options(valid, ref_sharp), unplaced, ref_sharp, 'clean_view')
        # 2) remaining measurements: nearby less-obstructed frames reached by validated chains.
        valid_rest = [m for m in valid if m.id in unplaced]
        options = {}                     # frame -> {measurement id: Placement}
        for m in valid_rest:
            pts = [[m.hook.x, m.hook.y], [m.housing.x, m.housing.y]]
            tried = ok_n = 0
            why = {'no_chain': 0, 'round_trip_error': 0, 'endpoint_outside_frame': 0, 'appearance_mismatch': 0}
            for f in self._candidates(m, ref_sharp):
                tried += 1
                mapped, ok, err, st = self.reg.map_validated(m.canvas_frame, f, pts, self.tol_frac)
                if mapped is None:
                    why['no_chain'] += 1
                    continue
                if not ok.all():
                    if not all(st.get('inside', [True])):
                        why['endpoint_outside_frame'] += 1
                    elif not all(st.get('cycle_ok', [False])):
                        why['round_trip_error'] += 1
                    else:
                        why['appearance_mismatch'] += 1
                    continue
                ok_n += 1
                options.setdefault(f, {})[m.id] = Placement(
                    m.id, [round(float(v), 1) for v in mapped[0]], [round(float(v), 1) for v in mapped[1]],
                    [round(float(e), 1) for e in err], st.get('links', 0), st.get('local_moves', 0),
                    st.get('global_moves', 0), 'reading_frame' if f == m.canvas_frame else 'registered')
            self.log.append({'measurement': m.id, 'stage': 'nearby_view', 'views_tried': tried, 'views_ok': ok_n,
                             'rejected_because': why})
        canvases += self._cover(options, unplaced, ref_sharp, 'registered_view')
        # 3) Fallback: the reading frame itself (endpoints exactly as localised there).
        for m in valid:
            if m.id in unplaced:
                canvases.append(SceneCanvas(m.canvas_frame, m.canvas_t, 'reading_frame',
                                            self.clarity(m.canvas_frame, ref_sharp),
                                            [Placement(m.id, [m.hook.x, m.hook.y], [m.housing.x, m.housing.y],
                                                       [0.0, 0.0], 0, 0, 0, 'reading_frame')]))
        canvases.sort(key=lambda c: c.t)
        return canvases
