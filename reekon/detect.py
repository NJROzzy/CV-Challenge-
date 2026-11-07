"""Stage 2a: per-frame tape detection.

Classical strategy (CPU, no training): the blade of the tape measure in the footage is a long, narrow,
nearly straight DARK strip with bright tick marks/numbers, attached at one end to a much wider dark
housing. Per frame:

1. dark mask (V channel relative to the frame median), close (fills white numbers), open (removes thin
   cabinet gaps and edges);
2. seed = the longest connected component whose cross-section width is blade-like;
3. grow: add components lying on the seed's line (blade split by numbers, a hand, or blur);
4. cross-section profile along the line: blade-width bins vs housing-width bins (> ~2.3 x blade);
   the housing end is where the profile widens, the hook end is the opposite extreme;
5. evidence: blade length, bright-tick texture inside the strip, straightness, coverage.

All coordinates returned are full-resolution video pixels.
"""
from dataclasses import dataclass, field, asdict
import math

import cv2
import numpy as np

from .config import DETECTION_WIDTH
from .video import resize_to_width


@dataclass
class Endpoint:
    x: float
    y: float
    kind: str              # 'hook' or 'housing'
    visible: bool          # False when the end is cut off by the image border (true end not observed)

    @property
    def xy(self):
        return np.array([self.x, self.y])


@dataclass
class TapeObservation:
    frame_index: int
    t: float
    detector: str
    confidence: float
    angle_deg: float                      # blade direction in [0, 180)
    blade_width_px: float
    hook: Endpoint
    housing: Endpoint
    housing_detected: bool                # housing seen (wide end) rather than inferred
    evidence: dict = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)


def _odd(n):
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


class ClassicalTapeDetector:
    name = 'classical'

    def __init__(self, proc_width=DETECTION_WIDTH, dark_abs=90, dark_rel=0.45, min_confidence=0.25,
                 min_transitions=0.9, min_two_sided=0.4):
        # Strip thresholds, measured on 24 frames from the 4 videos: true blades >= 1.38
        # alternations per width (false lines <= 0.79); two-sided fraction true mostly 0.85-1.0
        # (a vegetation edge 0.17). Set conservatively between the two groups.
        self.min_transitions = min_transitions
        self.min_two_sided = min_two_sided
        self.proc_width = proc_width
        self.dark_abs = dark_abs
        self.dark_rel = dark_rel
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------ masks
    def dark_mask(self, small):
        v = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 2]
        ref = float(np.median(v))
        dark = (v < min(self.dark_abs, self.dark_rel * ref)).astype(np.uint8) * 255
        W = small.shape[1]
        ell = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(k), _odd(k)))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, ell(0.019 * W))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, ell(0.014 * W))
        return dark, v, ref

    # ------------------------------------------------------------------ geometry helpers
    @staticmethod
    def _frame(points, center, direction):
        rel = points - center
        normal = np.array([-direction[1], direction[0]])
        return rel @ direction, rel @ normal

    @staticmethod
    def _profile(t, s, bin_px):
        """Cross-section extent of the mask per bin along the line."""
        lo = t.min()
        idx = ((t - lo) // bin_px).astype(int)
        n = idx.max() + 1
        smin = np.full(n, np.inf)
        smax = np.full(n, -np.inf)
        np.minimum.at(smin, idx, s)
        np.maximum.at(smax, idx, s)
        extent = np.where(np.isfinite(smin), smax - smin + 1, 0.0)
        centers = lo + (np.arange(n) + 0.5) * bin_px
        return centers, extent

    @staticmethod
    def _robust_line(pts, max_points=3000):
        """Huber line fit (cv2.fitLine) on a fixed random subsample: the fit cost is linear in the
        number of mask pixels (~10k+ for a near blade), the line itself needs far fewer.
        (A hand-rolled weighted-PCA IRLS was tried and rejected: ~2 deg bias on long blades.)"""
        if len(pts) > max_points:
            pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
        line = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        return np.array(line[2:], dtype=np.float64), np.array(line[:2], dtype=np.float64)

    def _component_stats(self, pts, bin_px):
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        d = vt[0]
        t, s = self._frame(pts, c, d)
        _, extent = self._profile(t, s, bin_px)
        filled = extent[extent > 0]
        width = float(np.median(filled)) if len(filled) else 0.0
        return c, d, float(t.max() - t.min()), width

    # ------------------------------------------------------------------ detection
    def detect(self, image, frame_index=-1, t=0.0):
        small, scale = resize_to_width(image, self.proc_width)
        H, W = small.shape[:2]
        mask, v, ref = self.dark_mask(small)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        bin_px = 3.0
        comps = {}
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 0.0004 * W * H:
                continue
            x, y, bw, bh = stats[i, :4]
            ys, xs = np.nonzero(labels[y:y + bh, x:x + bw] == i)
            comps[i] = np.column_stack([xs + x, ys + y]).astype(np.float64)
        if not comps:
            return None

        # Seed: longest component with a blade-like cross-section.
        best = None
        for i, pts in comps.items():
            c, d, length, width = self._component_stats(pts, bin_px)
            if not (0.012 * W <= width <= 0.07 * W) or length < max(6 * width, 0.12 * W):
                continue
            if best is None or length > best[3]:
                best = (i, c, d, length, width)
        if best is None:
            return None
        _, c, d, _, w = best

        # Grow along the line: blade pieces split by numbers / hands / blur, and the housing.
        members = {best[0]}
        for _ in range(3):
            core = np.vstack([comps[i] for i in members])
            t_core, _ = self._frame(core, c, d)
            lo, hi = t_core.min(), t_core.max()
            for i, pts in comps.items():
                if i in members:
                    continue
                t_i, s_i = self._frame(pts, c, d)
                on_line = np.mean(np.abs(s_i) <= 1.5 * w)
                gap = max(lo - t_i.max(), t_i.min() - hi, 0)
                # Blade pieces lie along the line; a housing split off from the blade (e.g. by
                # a bright LCD/hand) is wide but straddles the line right at one end.
                straddles = np.mean(np.abs(s_i) <= 0.6 * w) >= 0.05 and np.ptp(s_i) <= 8 * w
                if (on_line >= 0.5 and gap <= 0.3 * W) or (straddles and gap <= 3 * w):
                    members.add(i)
            pts = np.vstack([comps[i] for i in members])
            t_all, s_all = self._frame(pts, c, d)
            c, d = self._robust_line(pts[np.abs(s_all) <= 1.2 * w])

        pts = np.vstack([comps[i] for i in members])
        t_all, s_all = self._frame(pts, c, d)
        band = np.abs(s_all) <= 6 * w
        t_all, s_all = t_all[band], s_all[band]
        centers, extent = self._profile(t_all, s_all, bin_px)
        # Re-estimate blade width from narrow bins only (the housing would inflate it).
        narrow = extent[(extent > 0) & (extent <= 2.0 * w)]
        if len(narrow):
            w = float(np.median(narrow))
        is_housing = extent >= 2.0 * w

        # Housing: a run of wide bins starting at (or within a few bins of) one extreme of the tape.
        nb = len(extent)

        reach = max(4, int(15 * w / bin_px))  # housing must lie within ~15 blade widths of an extreme

        min_run = max(2, int(1.5 * w / bin_px))

        def housing_run(order):
            """Find housing support and its blade-facing entry (1-bin dips allowed).

            The longest wide run establishes which end contains the housing. A hand/LCD can
            split that housing into several runs: its entry is the innermost run that itself
            meets min_run, rather than the inner edge of the longest body fragment.
            """
            order = list(order)[:reach]
            best, cur, inner, dip = (0, None), 0, None, 0
            entry = None
            for k in order:
                if is_housing[k]:
                    cur, inner, dip = cur + 1, k, 0
                elif cur and dip == 0:
                    dip = 1
                    continue
                else:
                    cur, dip = 0, 0
                if cur >= min_run:
                    entry = inner
                if cur > best[0]:
                    best = (cur, inner)
            return best[0], entry

        run_hi, end_hi = housing_run(range(nb - 1, -1, -1))
        run_lo, end_lo = housing_run(range(nb))
        housing_detected = max(run_hi, run_lo) >= min_run
        housing_at_hi = run_hi >= run_lo

        t_min, t_max = t_all.min(), t_all.max()
        if housing_detected:
            # Endpoint B = blade/housing transition (where the blade enters the housing).
            if housing_at_hi:
                t_housing, t_hook = centers[end_hi] - bin_px / 2, t_min
            else:
                t_housing, t_hook = centers[end_lo] + bin_px / 2, t_max
        else:
            t_housing = t_max
            t_hook = t_min

        p_hook = c + d * t_hook
        p_housing = c + d * t_housing
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        occlusion = {}

        def to_border(p, along):
            """Distance from p to the image border walking along `along`."""
            ts = []
            for k, lim in ((0, W - 1), (1, H - 1)):
                if along[k] > 1e-9:
                    ts.append((lim - p[k]) / along[k])
                elif along[k] < -1e-9:
                    ts.append(-p[k] / along[k])
            return max(0.0, min(ts)) if ts else np.inf

        def end_seen(p, along, key):
            """True only if this is where the blade really ends: the blade does not run into the frame
            edge (a real end near the edge leaves a gap of background) and does not disappear behind a
            hand or sleeve."""
            # Measured (blade widths from the end to the border along the blade): ends cut off by
            # the frame edge 0.71-0.99, genuine hooks near the edge 1.18-1.30. Thin margin; an end
            # hidden behind over-exposed fingers is NOT caught (colour equals a white surface).
            if to_border(p, along) <= max(3.0, 1.1 * w):
                occlusion[key] = 'frame_edge'
                return False
            if self._occluded(hsv, p, along, w):
                occlusion[key] = 'hand_or_sleeve'
                return False
            return True

        hook_dir = -d if t_hook < t_housing else d
        hook_visible = end_seen(p_hook, hook_dir, 'hook')
        housing_visible = housing_detected or end_seen(p_housing, -hook_dir, 'housing')

        if not housing_detected:
            # No wide end seen. If exactly one end is cut off, the in-frame end is the hook
            # (the housing is the end held out of view); otherwise the labels are ambiguous.
            if not hook_visible and housing_visible:
                p_hook, p_housing = p_housing, p_hook
                hook_visible, housing_visible = True, False
            housing_visible = False  # never claim an unobserved housing end

        # Evidence ----------------------------------------------------------
        blade_len = float(np.linalg.norm(p_housing - p_hook))
        lo, hi = sorted([t_hook, t_housing])
        in_blade = (centers >= lo) & (centers <= hi)
        coverage = float(np.mean(extent[in_blade] > 0)) if in_blade.any() else 0.0
        texture = self._texture(v, ref, c, d, lo, hi, w)
        resid = s_all[(t_all >= lo) & (t_all <= hi) & (np.abs(s_all) <= 1.2 * w)]
        straight_rms = float(np.sqrt(np.mean(resid ** 2))) / w if len(resid) else 1.0

        length_score = min(1.0, blade_len / (0.35 * W))
        texture_score = 1.0 if 0.06 <= texture <= 0.55 else max(0.0, 1 - abs(texture - (0.06 if texture < 0.06 else 0.55)) / 0.06)
        straight_score = float(np.clip(1 - (straight_rms - 0.35) / 0.4, 0, 1))
        confidence = length_score * texture_score * math.sqrt(straight_score * coverage)
        if housing_detected:
            confidence = min(1.0, confidence * 1.15)
        if confidence < self.min_confidence:
            return None
        # Strip verification (rejects railings, vegetation edges, poles, shadows): a tape blade is a
        # dark strip darker than the surface on BOTH sides, with regularly alternating bright marks.
        transitions, two_sided = self._strip_evidence(v, ref, p_hook, p_housing, w)
        if transitions < self.min_transitions or two_sided < self.min_two_sided:
            return None

        if housing_detected:
            # A split housing can have a blade-width body fragment. The whole-blade
            # score can pass while B is on that body. Require local strip contrast
            # on the blade side of B before treating the entry itself as observed.
            _, entry_contrast = self._strip_evidence(
                v, ref, p_housing, p_housing + hook_dir * min(blade_len, 6 * w), w)
            if entry_contrast < self.min_two_sided:
                housing_visible = False
                occlusion['housing'] = 'entry_not_supported'

        inv = 1.0 / scale
        angle = (math.degrees(math.atan2(d[1], d[0])) + 180) % 180
        return TapeObservation(
            frame_index=frame_index, t=t, detector=self.name, confidence=round(float(confidence), 3),
            angle_deg=round(angle, 1), blade_width_px=round(w * inv, 1),
            hook=Endpoint(float(p_hook[0] * inv), float(p_hook[1] * inv), 'hook', bool(hook_visible)),
            housing=Endpoint(float(p_housing[0] * inv), float(p_housing[1] * inv), 'housing', bool(housing_visible)),
            housing_detected=bool(housing_detected),
            evidence={'blade_length_px': round(blade_len * inv, 1), 'tick_texture': round(texture, 3),
                      'straightness_rms_widths': round(straight_rms, 3), 'coverage': round(coverage, 3),
                      'length_score': round(length_score, 3), 'texture_score': round(texture_score, 3),
                      'straight_score': round(straight_score, 3),
                      'tick_transitions_per_width': round(transitions, 2),
                      'two_sided_dark_fraction': round(two_sided, 2),
                      'end_not_seen_because': occlusion})

    @staticmethod
    def _sample(img, pts):
        x = np.clip(np.round(pts[..., 0]).astype(int), 0, img.shape[1] - 1)
        y = np.clip(np.round(pts[..., 1]).astype(int), 0, img.shape[0] - 1)
        return img[y, x]

    def _occluded(self, hsv, p, along, w):
        """Is the area just beyond this end mostly hand/sleeve (skin-to-orange hues, saturated)?
        The small brass hook tab is far smaller than the sampled patch, so it does not trigger this."""
        n = np.array([-along[1], along[0]])
        pts = np.array([p + along * a + n * b for a in np.linspace(0.7 * w, 3.0 * w, 6)
                        for b in np.linspace(-1.2 * w, 1.2 * w, 5)])
        h, s_, v_ = (self._sample(hsv, pts)[:, k].astype(int) for k in range(3))
        skin = ((h <= 25) | (h >= 165)) & (s_ >= 50) & (v_ >= 60)
        return float(np.mean(skin)) >= 0.35

    def _strip_evidence(self, v, ref, a, b, w):
        """(bright/dark alternations per blade width along the centre line,
             fraction of the strip darker than the surface on both sides)."""
        L = float(np.linalg.norm(b - a))
        if L < 4 * w:
            return 0.0, 0.0
        d = (b - a) / L
        n = np.array([-d[1], d[0]])
        P = a + np.arange(0, L, 2.0)[:, None] * d
        centre = self._sample(v, P).astype(float)
        transitions = float(np.abs(np.diff((centre > 0.55 * ref).astype(int))).sum() / (L / w))
        core = np.min([self._sample(v, P + k * w * n) for k in (-0.25, 0.0, 0.25)], axis=0).astype(float)
        left = np.median([self._sample(v, P - k * w * n) for k in (1.0, 1.3, 1.6)], axis=0)
        right = np.median([self._sample(v, P + k * w * n) for k in (1.0, 1.3, 1.6)], axis=0)
        H, W = v.shape
        inside = (P[:, 0] > 2 * w) & (P[:, 0] < W - 2 * w) & (P[:, 1] > 2 * w) & (P[:, 1] < H - 2 * w)
        if not inside.any():
            return transitions, 0.0
        both = ((left - core) > 0.25 * ref) & ((right - core) > 0.25 * ref)
        return transitions, float(np.mean(both[inside]))

    @staticmethod
    def _texture(v, ref, c, d, lo, hi, w):
        """Fraction of bright (tick/number) pixels inside the central strip of the blade."""
        n = np.array([-d[1], d[0]])
        ts = np.arange(lo, hi, 1.0)
        if len(ts) < 5:
            return 0.0
        offsets = np.linspace(-0.3 * w, 0.3 * w, 5)
        pts = c + ts[:, None, None] * d + offsets[None, :, None] * n
        xs = np.clip(np.round(pts[..., 0]).astype(int), 0, v.shape[1] - 1)
        ys = np.clip(np.round(pts[..., 1]).astype(int), 0, v.shape[0] - 1)
        vals = v[ys, xs]
        return float(np.mean(vals > 0.55 * ref))
