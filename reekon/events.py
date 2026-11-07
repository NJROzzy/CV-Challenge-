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
from .endpoints import EndpointEstimate, EndpointLocalizer, mode_cluster  # compatibility exports


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
    status: str = 'candidate'     # 'validated' | 'uncertain' | 'rejected' after validate()
    reasons: list = field(default_factory=list)
    blade_length_frac: float = 0.0   # median visible blade length / shorter frame side, in the reading

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


def enforce_identity(episode, max_dt=0.5, diag=None):
    """Keep hook/housing labels consistent over an episode.

    Consecutive observations are linked by whether their ends keep or exchange labels (whichever
    pairing moves the ends less). That gives each observation a parity relative to the first one in
    its chain; the chain's orientation is then set by majority vote of the observations that
    actually SAW the wide housing. Observations that disagree with the vote are relabelled if their
    labels were only inferred (no housing seen), or marked as conflicting (excluded from endpoint
    voting) if the detector saw a housing at the other end. Returns the number of relabelled and
    conflicting observations."""
    relabelled = conflicts = 0
    chains, cur = [], []
    for o in episode:
        if cur:
            p = cur[-1][0]
            same = np.linalg.norm(o.hook.xy - p.hook.xy) + np.linalg.norm(o.housing.xy - p.housing.xy)
            swap = np.linalg.norm(o.hook.xy - p.housing.xy) + np.linalg.norm(o.housing.xy - p.hook.xy)
            jump = diag is not None and min(same, swap) > 0.3 * diag
            if o.t - p.t > max_dt or jump:
                chains.append(cur)
                cur = []
            else:
                cur.append((o, cur[-1][1] ^ (swap < 0.7 * same)))
                continue
        cur.append((o, False))
    if cur:
        chains.append(cur)
    for chain in chains:
        vote = sum((1 if par else -1) for o, par in chain if o.housing_detected)
        flip_parity = vote > 0          # True: observations with parity=True carry the right labels
        for o, par in chain:
            if par == flip_parity:
                continue
            if o.housing_detected:
                o.evidence['identity_conflict'] = True
                conflicts += 1
            else:
                o.hook, o.housing = o.housing, o.hook
                o.hook.kind, o.housing.kind = 'hook', 'housing'
                o.hook.visible = False              # this end was never established as the real tip
                o.housing.visible = False
                o.evidence['identity_relabelled'] = True
                relabelled += 1
    return relabelled, conflicts


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


class EventBuilder:
    def __init__(self, registrar, sharpness, frame_size, cluster_radius_frac=0.04,
                 min_support=3, min_hook_confidence=0.5):
        self.reg = registrar
        self.sharpness = sharpness          # frame index -> variance of Laplacian
        self.W, self.H = frame_size
        self.radius = cluster_radius_frac * math.hypot(self.W, self.H)
        self.min_hook_confidence = min_hook_confidence
        self.endpoints = EndpointLocalizer(registrar, self.radius, min_support)
        self.log = []

    def _inside(self, p, margin=0.0):
        return margin <= p[0] <= self.W - 1 - margin and margin <= p[1] <= self.H - 1 - margin

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
        ranked = [o for o in self._rank(phase) if o.housing.visible][:max_candidates]
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
        diag = math.hypot(self.W, self.H)
        self.identity = {'relabelled': 0, 'conflicts': 0}
        for ep_id, episode in enumerate(episodes):
            r, c = enforce_identity(episode, diag=diag)
            self.identity['relabelled'] += r
            self.identity['conflicts'] += c
            usable = [o for o in episode if not o.evidence.get('identity_conflict')]
            phases = housing_phases(usable)
            all_hooks = [o for o in usable if o.hook.visible and o.confidence >= self.min_hook_confidence]
            for k, phase in enumerate(phases):
                # The hook may be re-placed between readings (e.g. the next door), so a reading only
                # uses hook sightings between the neighbouring readings, not the whole episode.
                lo = phases[k - 1][-1].t if k > 0 else -np.inf
                hi = phases[k + 1][0].t if k + 1 < len(phases) else np.inf
                hooks = [o for o in all_hooks if lo <= o.t <= hi]
                canvas_obs = self._pick_canvas(phase)
                canvas = canvas_obs.frame_index
                housing, _ = self.endpoints.estimate('housing', phase, canvas)
                hook, k_used = self.endpoints.estimate('hook', hooks, canvas, canvas_obs)
                notes = []
                # Final reference frame: its own housing must agree with the consensus; prefer one
                # that directly shows BOTH endpoints (no long registration chain needed).
                if housing.status != 'missing':
                    chosen, shows_hook = self._consistent_canvas(phase, canvas, housing)
                    if chosen is None:
                        notes.append('no reading frame matched the housing consensus; using the sharpest frame')
                    elif chosen.frame_index != canvas:
                        canvas_obs, canvas = chosen, chosen.frame_index
                        housing, _ = self.endpoints.estimate('housing', phase, canvas)
                        hook, k_used = self.endpoints.estimate('hook', hooks, canvas, canvas_obs)
                    if shows_hook and hook.status == 'observed':
                        notes.append('reference frame directly shows both endpoints')
                context = None
                if hook.status == 'registered':
                    self.endpoints.snap_to_tape(hook, canvas_obs, notes)
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
                m.blade_length_frac = round(float(np.median(
                    [np.linalg.norm(o.hook.xy - o.housing.xy) for o in phase])) / min(self.W, self.H), 3)
                results.append(m)
        results = self._merge(results)
        for m in results:
            self.validate(m)
        # Validated measurements are numbered 1..n; candidates keep separate numbering (C1, C2, ...).
        n = c = 0
        for m in results:
            if m.status == 'validated':
                n += 1
                m.id = n
            else:
                c += 1
                m.id = c
        return results

    def validate(self, m, min_confidence=0.4, min_blade_frac=0.2):
        """Decide whether a reading is a confirmed measurement. Everything else is kept as a
        diagnostic candidate with its reasons; nothing is silently dropped."""
        reasons = []
        if m.hook.status == 'missing' and m.housing.status == 'missing':
            reasons.append('no endpoint could be localised')
        elif m.hook.status == 'missing':
            reasons.append('hook end not localised (never seen clearly, or not mappable)')
        elif m.housing.status == 'missing':
            reasons.append('housing end not localised')
        if m.blade_length_frac < min_blade_frac:
            reasons.append(f'blade barely extended ({m.blade_length_frac:.2f} of frame) - carrying/repositioning?')
        if m.confidence < min_confidence and not reasons:
            reasons.append(f'low confidence {m.confidence:.2f} < {min_confidence}')
        if not reasons:
            m.status = 'validated'
        elif m.hook.status == 'missing' and m.housing.status == 'missing' or m.blade_length_frac < min_blade_frac:
            m.status = 'rejected'
        else:
            m.status = 'uncertain'
        m.reasons = reasons
        return m

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
