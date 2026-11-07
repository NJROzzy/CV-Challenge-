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
    kind: str                     # 'registered_view' or 'reading_frame' (fallback)
    clarity: float
    placements: list = field(default_factory=list)

    def as_dict(self):
        return {'frame': self.frame, 't': self.t, 'kind': self.kind, 'clarity': round(self.clarity, 3),
                'placements': [p.__dict__ for p in self.placements]}


class SceneBuilder:
    def __init__(self, registrar, sharpness, clutter, times, window_s=10.0, candidates_per_measurement=30,
                 min_separation_s=0.4, tol_frac=0.02):
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

    def build(self, measurements):
        valid = [m for m in measurements if m.status == 'validated']
        if not valid:
            return []
        ref_sharp = float(np.percentile(list(self.sharpness.values()), 90))
        options = {}                     # frame -> {measurement id: Placement}
        for m in valid:
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
            self.log.append({'measurement': m.id, 'views_tried': tried, 'views_ok': ok_n, 'rejected_because': why})

        # Greedy cover: the frame placing the most still-unplaced measurements, ties -> clearer frame.
        canvases, unplaced = [], {m.id for m in valid}
        while unplaced:
            best = max(options, default=None, key=lambda f: (len(unplaced & options[f].keys()),
                                                             self.clarity(f, ref_sharp)))
            if best is None or not unplaced & options[best].keys():
                break
            ids = sorted(unplaced & options[best].keys())
            placements = [options[best][i] for i in ids]
            kind = 'registered_view' if any(pl.source == 'registered' for pl in placements) else 'reading_frame'
            canvases.append(SceneCanvas(best, round(self.times[best], 3), kind,
                                        self.clarity(best, ref_sharp), placements))
            unplaced -= set(ids)
        # Fallback: the reading frame itself (endpoints exactly as localised there).
        for m in valid:
            if m.id in unplaced:
                canvases.append(SceneCanvas(m.canvas_frame, m.canvas_t, 'reading_frame',
                                            self.clarity(m.canvas_frame, ref_sharp),
                                            [Placement(m.id, [m.hook.x, m.hook.y], [m.housing.x, m.housing.y],
                                                       [0.0, 0.0], 0, 0, 0, 'reading_frame')]))
        canvases.sort(key=lambda c: c.t)
        return canvases
