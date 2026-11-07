"""Stage 2b/3/4b: measurement events, endpoint localisation and reference-frame (canvas) selection.

Temporal structure observed in PoV measuring footage:
  * an *episode* is a continuous run of tape detections with a consistent blade direction;
  * a *reading* (one measurement) is a run inside an episode where the housing is visible:
    the user holds the housing at the second endpoint while the hook stays fixed at the first;
  * the hook is often out of view during the reading (camera pointed at the housing/display) and
    is observed earlier or later in the same episode.

Endpoints are therefore estimated from *many* frames: every observation is mapped into one
canvas frame by registration, and the endpoint is the densest cluster (mode) of the mapped
points. Approach motion, blur and single-frame errors fall outside the mode. Nothing is
invented: an endpoint without enough agreeing observations is reported as missing.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from .register import map_points


@dataclass
class EndpointEstimate:
    kind: str                     # 'hook' or 'housing'
    status: str                   # 'observed' (seen in the canvas frame), 'registered', 'missing'
    x: float = None
    y: float = None
    confidence: float = 0.0
    support: int = 0              # observations agreeing with the estimate
    candidates: int = 0           # observations considered
    spread_px: float = None       # RMS distance of agreeing observations from the estimate
    source_frames: list = field(default_factory=list)
    alternatives: list = field(default_factory=list)  # other clusters with real support

    def as_dict(self):
        return dict(self.__dict__)


@dataclass
class Measurement:
    id: int
    start_t: float
    end_t: float
    canvas_frame: int
    canvas_t: float
    hook: EndpointEstimate
    housing: EndpointEstimate
    confidence: float
    angle_deg: float
    notes: list = field(default_factory=list)
    context_frame: int = None     # frame to mosaic in when an endpoint lies outside the canvas

    def as_dict(self):
        d = {k: v for k, v in self.__dict__.items() if k not in ('hook', 'housing')}
        d['hook'] = self.hook.as_dict()
        d['housing'] = self.housing.as_dict()
        return d


def _angle_diff(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def split_episodes(observations, max_gap=0.8, max_turn=30.0):
    episodes, cur = [], []
    for o in observations:
        if cur:
            ref = float(np.median([p.angle_deg for p in cur[-5:]]))
            if o.t - cur[-1].t > max_gap or _angle_diff(o.angle_deg, ref) > max_turn:
                episodes.append(cur)
                cur = []
        cur.append(o)
    if cur:
        episodes.append(cur)
    return episodes


def housing_phases(episode, max_gap=0.8, min_duration=0.8, min_obs=5):
    phases, cur = [], []
    for o in episode:
        if not o.housing_detected:
            continue
        if cur and o.t - cur[-1].t > max_gap:
            phases.append(cur)
            cur = []
        cur.append(o)
    if cur:
        phases.append(cur)
    return [p for p in phases if p[-1].t - p[0].t >= min_duration and len(p) >= min_obs]


def mode_cluster(points, radius):
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


class EventBuilder:
    def __init__(self, registrar, sharpness, frame_size, cluster_radius_frac=0.04,
                 min_support=3, min_hook_confidence=0.5):
        self.reg = registrar
        self.sharpness = sharpness          # frame index -> variance of Laplacian
        self.W, self.H = frame_size
        self.radius = cluster_radius_frac * math.hypot(self.W, self.H)
        # Hook sightings must map within this distance of the visible tape line. Measured on
        # Cabinet-1.MOV: wide-angle registration error up to ~200 px across large pans, while a
        # different door's hook lands ~640 px away.
        self.line_tolerance = 3 * self.radius
        self.min_support = min_support
        self.min_hook_confidence = min_hook_confidence
        self.log = []

    def _inside(self, p, margin=0.0):
        return margin <= p[0] <= self.W - 1 - margin and margin <= p[1] <= self.H - 1 - margin

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

    def _estimate(self, kind, observations, canvas, canvas_obs=None):
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

    def _snap_to_tape(self, hook, canvas_obs, notes):
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

    def _rank(self, phase):
        """Reading frames by preference: sharp, near the middle of the reading, confident."""
        mid = (phase[0].t + phase[-1].t) / 2
        span = max(phase[-1].t - phase[0].t, 1e-6)
        sharp = np.array([self.sharpness.get(o.frame_index, 0.0) for o in phase])
        sharp = sharp / (sharp.max() or 1)
        central = 1 - np.abs([(o.t - mid) / span for o in phase])   # 1 at the middle, 0.5 at the ends
        score = sharp * central * np.array([o.confidence for o in phase])
        return [phase[i] for i in np.argsort(-score)]

    def _pick_canvas(self, phase):
        return self._rank(phase)[0]

    def _consistent_canvas(self, phase, provisional, housing, max_candidates=15):
        """Best-ranked frame whose own housing detection lies at the reading's consensus position,
        preferring frames that also show the hook. Rejects frames from the approach (housing still
        moving) or the departure (camera moving away). Returns (obs, shows_hook) or (None, False)."""
        ranked = self._rank(phase)[:max_candidates]
        ranked.sort(key=lambda o: not o.hook.visible)  # stable: hook-visible frames first
        for o in ranked:
            r = self.reg.register(provisional, o.frame_index)
            if not r.ok:
                continue
            hs = map_points(r.H, [housing.consensus_xy])[0]
            # Frames that also show the hook are worth more (no chain needed) but are taken with the
            # camera tilted/panned away, where registration error is larger: allow a full radius.
            tol = self.radius if o.hook.visible else 0.5 * self.radius
            if np.linalg.norm(hs - o.housing.xy) <= tol:
                return o, o.hook.visible
        return None, False

    def build(self, observations):
        episodes = split_episodes(observations)
        self.log.append({'episodes': [(e[0].t, e[-1].t, len(e)) for e in episodes]})
        results = []
        for ep_id, episode in enumerate(episodes):
            phases = housing_phases(episode)
            all_hooks = [o for o in episode if o.hook.visible and o.confidence >= self.min_hook_confidence]
            for k, phase in enumerate(phases):
                # The hook may be re-placed between readings (e.g. the next door), so a reading only
                # uses hook sightings between the neighbouring readings, not the whole episode.
                lo = phases[k - 1][-1].t if k > 0 else -np.inf
                hi = phases[k + 1][0].t if k + 1 < len(phases) else np.inf
                hooks = [o for o in all_hooks if lo <= o.t <= hi]
                canvas_obs = self._pick_canvas(phase)
                canvas = canvas_obs.frame_index
                housing, _ = self._estimate('housing', phase, canvas)
                hook, k_used = self._estimate('hook', hooks, canvas, canvas_obs)
                notes = []
                # Final reference frame: its own housing must agree with the consensus; prefer one
                # that directly shows BOTH endpoints (no long registration chain needed).
                if housing.status != 'missing':
                    chosen, shows_hook = self._consistent_canvas(phase, canvas, housing)
                    if chosen is None:
                        notes.append('no reading frame matched the housing consensus; using the sharpest frame')
                    elif chosen.frame_index != canvas:
                        canvas_obs, canvas = chosen, chosen.frame_index
                        housing, _ = self._estimate('housing', phase, canvas)
                        hook, k_used = self._estimate('hook', hooks, canvas, canvas_obs)
                    if shows_hook and hook.status == 'observed':
                        notes.append('reference frame directly shows both endpoints')
                context = None
                if hook.status == 'registered':
                    self._snap_to_tape(hook, canvas_obs, notes)
                if hook.status == 'missing':
                    notes.append('hook end not observed with enough agreeing frames for this reading')
                elif not self._inside((hook.x, hook.y)):
                    # Hook lies outside the canvas frame: mosaic in a sharp, well-registered hook frame.
                    def ctx_score(m):
                        o, _, r = m
                        return self.sharpness.get(o.frame_index, 0) * o.confidence / (1 + r.links / 20)
                    context = max(k_used, key=ctx_score)[0].frame_index
                    notes.append('hook end lies outside the canvas frame; shown via registered mosaic')
                if hook.alternatives:
                    notes.append(f'{len(hook.alternatives)} other hook position cluster(s) seen for this reading')
                confs = [e.confidence for e in (hook, housing)]
                m = Measurement(id=0, start_t=round(phase[0].t, 3), end_t=round(phase[-1].t, 3),
                                canvas_frame=canvas, canvas_t=round(canvas_obs.t, 3), hook=hook, housing=housing,
                                confidence=round(float(np.prod(confs)) ** 0.5 if all(confs) else 0.0, 3),
                                angle_deg=float(np.median([o.angle_deg for o in phase])), notes=notes,
                                context_frame=context)
                m.episode = ep_id
                results.append(m)
        results = self._merge(results)
        for i, m in enumerate(results, 1):
            m.id = i
        return results

    def _merge(self, ms):
        """Readings in the same episode at the same physical endpoints are one measurement
        (e.g. the camera looked away from the housing and back)."""
        merged = []
        for m in ms:
            prev = merged[-1] if merged else None
            if prev is not None and prev.episode == m.episode and 'missing' not in (m.hook.status, m.housing.status, prev.hook.status, prev.housing.status):
                r = self.reg.register(m.canvas_frame, prev.canvas_frame)
                if r.ok:
                    hk, hs = map_points(r.H, [[m.hook.x, m.hook.y], [m.housing.x, m.housing.y]])
                    if np.linalg.norm(hk - [prev.hook.x, prev.hook.y]) <= self.radius and \
                            np.linalg.norm(hs - [prev.housing.x, prev.housing.y]) <= self.radius:
                        prev.end_t = m.end_t
                        prev.notes.append(f'merged with reading {m.start_t:.1f}-{m.end_t:.1f}s (same endpoints)')
                        if m.confidence > prev.confidence:  # keep the better-supported view
                            m.start_t, m.notes = prev.start_t, prev.notes
                            m.episode = prev.episode
                            merged[-1] = m
                        continue
            merged.append(m)
        return merged
