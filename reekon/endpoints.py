"""Endpoint consensus in full-resolution reference-frame pixels.

Registration supplies source-to-reference transforms. This module votes sightings,
keeps directly observed coordinates, and applies the existing straight-blade prior.
It does not split events, classify measurements, or choose rendering canvases.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detect import TapeObservation
from .register import map_points


@dataclass
class EndpointEstimate:
    kind: str                     # 'hook' or 'housing'
    status: str                   # 'observed' (seen in the canvas frame), 'registered', 'missing'
    x: float | None = None
    y: float | None = None
    confidence: float = 0.0
    support: int = 0              # observations agreeing with the estimate
    candidates: int = 0           # observations considered
    spread_px: float | None = None       # RMS distance of agreeing observations from the estimate
    source_frames: list[int] = field(default_factory=list)
    alternatives: list = field(default_factory=list)  # other clusters with real support

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def mode_cluster(points, radius: float) -> tuple[np.ndarray, np.ndarray, list]:
    """Densest cluster: the point with most neighbours within `radius`, refined by the median of
    those neighbours. Returns (center, member mask, other clusters as [(center, count)])."""
    P = np.asarray(points, dtype=np.float64)
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    counts = (D <= radius).sum(1)
    i = int(np.argmax(counts))
    members = D[i] <= radius
    center = np.median(P[members], axis=0)
    members = np.linalg.norm(P - center, axis=1) <= radius
    others, rest = [], ~members
    while rest.sum() >= 2:
        sub = np.where(rest)[0]
        c = (D[np.ix_(sub, sub)] <= radius).sum(1)
        k = sub[int(np.argmax(c))]
        m = rest & (D[k] <= radius)
        if m.sum() < 2:
            break
        others.append((np.median(P[m], axis=0), int(m.sum())))
        rest &= ~m
    return center, members, others


class EndpointLocalizer:
    def __init__(self, registrar, radius: float, min_support: int):
        self.reg = registrar
        self.radius = radius
        self.min_support = min_support
        # Footage calibration: ~200 px registration error vs ~640 px between doors.
        self.line_tolerance = 3 * radius

    def _map_all(self, observations, kind, canvas):
        mapped = []
        for o in observations:
            ep = getattr(o, kind)
            r = self.reg.register(o.frame_index, canvas)
            if not r.ok:
                continue
            q = map_points(r.H, [ep.xy])[0]
            mapped.append((o, q, r))
        return mapped

    def _on_tape_line(self, mapped, canvas_obs):
        """Keep mapped hook points consistent with the straight blade seen in the canvas frame:
        on the hook side of the housing and within `line_tolerance` of the line's extension.
        Removes sightings of a different hook placement (e.g. the neighbouring door)."""
        a, b = canvas_obs.housing.xy, canvas_obs.hook.xy
        u = (b - a) / (np.linalg.norm(b - a) or 1)
        keep = []
        for m in mapped:
            rel = m[1] - a
            along = float(np.dot(rel, u))
            perp = float(abs(rel[0] * u[1] - rel[1] * u[0]))
            if along > 0 and perp <= self.line_tolerance:
                keep.append(m)
        return keep

    def estimate(self, kind: str, observations: list[TapeObservation], canvas: int,
                 canvas_obs: TapeObservation | None = None) -> tuple[EndpointEstimate, list]:
        est = EndpointEstimate(kind, 'missing', candidates=len(observations))
        mapped = self._map_all(observations, kind, canvas)
        if canvas_obs is not None:
            before = len(mapped)
            mapped = self._on_tape_line(mapped, canvas_obs)
            est.rejected_off_tape_line = before - len(mapped)
        if len(mapped) < self.min_support:
            return est, mapped
        pts = np.array([q for _, q, _ in mapped])
        center, members, others = mode_cluster(pts, self.radius)
        support = int(members.sum())
        if support < self.min_support:
            return est, mapped
        used = [m for m, keep in zip(mapped, members) if keep]
        direct = any(o.frame_index == canvas for o, _, _ in used)
        spread = float(np.sqrt(np.mean(np.sum((pts[members] - center) ** 2, axis=1))))
        weakest = min((r.min_inliers for _, _, r in used if r.links), default=None)
        # Confidence: agreement (support fraction), tightness, and how many observations agree.
        frac = support / len(mapped)
        tight = max(0.0, 1 - spread / self.radius)
        count = min(1.0, support / 10)
        conf = frac * (0.5 + 0.5 * tight) * (0.5 + 0.5 * count)
        if direct:
            conf = min(1.0, conf + 0.15)
        est.status = 'observed' if direct else 'registered'
        est.consensus_xy = [round(float(center[0]), 1), round(float(center[1]), 1)]  # cluster median
        if direct:
            # Seen in the reference frame itself: draw exactly what was detected there. The other
            # (registered) observations only vote on agreement/confidence; registration error in a
            # moving frame must not move a directly observed point.
            own = next(q for (o, q, _) in used if o.frame_index == canvas)
            center = own
        est.x, est.y = float(center[0]), float(center[1])
        est.confidence = round(conf, 3)
        est.support = support
        est.spread_px = round(spread, 1)
        est.source_frames = sorted({o.frame_index for o, _, _ in used})
        est.alternatives = [{'x': float(c[0]), 'y': float(c[1]), 'support': n} for c, n in others
                            if n >= self.min_support]
        est.weakest_link_inliers = weakest
        return est, used

    def snap_to_tape(self, hook: EndpointEstimate, canvas_obs: TapeObservation, notes: list[str]) -> None:
        """The blade is physically straight and visible in the canvas frame, so the hook lies on the
        extension of that line. Registration across large camera motion (wide-angle distortion,
        parallax) mostly errs perpendicular to it; remove that component, keep the along-tape one."""
        a, b = canvas_obs.housing.xy, canvas_obs.hook.xy
        u = (b - a) / (np.linalg.norm(b - a) or 1)
        p = np.array([hook.x, hook.y])
        q = a + np.dot(p - a, u) * u
        offset = float(np.linalg.norm(p - q))
        hook.registered_xy = [round(float(p[0]), 1), round(float(p[1]), 1)]
        hook.snap_offset_px = round(offset, 1)
        if np.dot(p - a, u) <= 0 or offset > self.line_tolerance:
            notes.append(f'registered hook is {offset:.0f}px off the visible tape line; not snapped')
            hook.confidence = round(hook.confidence * 0.5, 3)
            return
        hook.x, hook.y = float(q[0]), float(q[1])
        notes.append(f'hook snapped {offset:.0f}px onto the visible tape line (straight-blade prior)')
