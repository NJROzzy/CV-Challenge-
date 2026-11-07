"""Stage 4a: frame registration (map image points between sampled frames of the same video).

Cabinets/walls are large and nearly textureless and the PoV lens is wide-angle, so direct
descriptor matching between distant viewpoints is unreliable (measured on Cabinet-1.MOV: 0-17
RANSAC inliers between frames 2-4 s apart). Instead:

1. track corners frame-to-frame with pyramidal Lucas-Kanade (forward-backward checked) between
   consecutive sampled frames, and fit a RANSAC homography per link;
2. compose links to reach distant frames;
3. refine: warp the source into the destination with the chained estimate and re-track to remove
   accumulated drift.

The tape, housing and hands move with the person, not the scene, so they are masked out.
A homography is exact only for a plane (e.g. a cabinet face) under an undistorted camera; lens
distortion and parallax leave residual error, which is reported, not hidden.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .video import resize_to_width


@dataclass
class Registration:
    ok: bool
    H: np.ndarray          # full-resolution src -> dst homography (identity when src == dst)
    links: int             # chained frame-to-frame links (0 = same frame)
    min_inliers: int       # weakest link
    refine_inliers: int    # inliers of the final drift-removal step (0 if not refined)

    def as_dict(self):
        return {'ok': self.ok, 'links': self.links, 'min_inliers': self.min_inliers,
                'refine_inliers': self.refine_inliers}


class Registrar:
    def __init__(self, work_width=640, min_inliers=12):
        self.work_width = work_width
        self.min_inliers = min_inliers
        self.frames = {}        # frame index -> (gray, feature mask, scale)
        self._links = {}
        self._pairs = {}
        self.clutter = {}
        self._sift = {}
        self._sift_detector = cv2.SIFT_create(1500)
        self.bridges_tried = self.bridges_ok = 0
        self.min_patch_ncc = 0.5

    # ------------------------------------------------------------------ frame store
    def add(self, frame_index, image, observation=None):
        small, scale = resize_to_width(image, self.work_width)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, np.uint8)
        if observation is not None:
            a = tuple(int(v) for v in observation.hook.xy * scale)
            b = tuple(int(v) for v in observation.housing.xy * scale)
            w = max(2, int(observation.blade_width_px * scale))
            cv2.line(mask, a, b, 0, 4 * w)
            cv2.circle(mask, b, 8 * w, 0, -1)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        skin = cv2.inRange(hsv, (0, 60, 70), (22, 200, 255))
        mask[cv2.dilate(skin, np.ones((9, 9), np.uint8)) > 0] = 0
        self.frames[frame_index] = (gray, mask, scale)
        # Share of the frame covered by tape/housing/hands: used to pick unobstructed scene views.
        self.clutter[frame_index] = float(np.mean(mask == 0))

    # ------------------------------------------------------------------ core estimation
    def _track(self, g0, m0, g1):
        p0 = cv2.goodFeaturesToTrack(g0, 800, 0.005, 7, mask=m0, blockSize=7)
        if p0 is None or len(p0) < self.min_inliers:
            return None, 0
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(25, 25), maxLevel=4)
        back, st_b, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, winSize=(25, 25), maxLevel=4)
        ok = (st.ravel() == 1) & (st_b.ravel() == 1) & (np.linalg.norm(back - p0, axis=2).ravel() < 1.0)
        if ok.sum() < self.min_inliers:
            return None, int(ok.sum())
        H, inl = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 2.0)
        n = int(inl.sum()) if inl is not None else 0
        if H is None or n < self.min_inliers:
            return None, n
        keep = inl.ravel().astype(bool)
        self._last_pairs = (p0[ok][keep].reshape(-1, 2), p1[ok][keep].reshape(-1, 2))
        return H, n

    def _link(self, a, b):
        key = (a, b)
        if key not in self._links:
            ga, ma, _ = self.frames[a]
            gb, _, _ = self.frames[b]
            self._last_pairs = None
            self._links[key] = self._track(ga, ma, gb)
            self._pairs[key] = self._last_pairs   # inlier correspondences, for local point motion
        return self._links[key]

    def _chain(self, src, dst):
        """Consecutive-frame links from src to dst (skipping up to two failed frames), or None."""
        keys = sorted(self.frames)
        i, j = keys.index(src), keys.index(dst)
        step = 1 if j > i else -1
        lo, hi = min(i, j), max(i, j)
        links, k = [], i
        while k != j:
            advanced = False
            for hop in (1, 2, 3):
                n = k + step * hop
                if not lo <= n <= hi:
                    break
                Hl, inl = self._link(keys[k], keys[n])
                if Hl is not None:
                    links.append((keys[k], keys[n], Hl, inl))
                    k, advanced = n, True
                    break
            if not advanced:
                # Keyframe bridge: fast head motion can defeat frame-to-frame tracking for several
                # samples; try descriptor matching (SIFT + RANSAC) to a frame up to ~1.5 s ahead.
                for hop in range(2, 16):
                    n = k + step * hop
                    if not lo <= n <= hi:
                        break
                    Hl, inl = self._bridge(keys[k], keys[n])
                    if Hl is not None:
                        links.append((keys[k], keys[n], Hl, inl))
                        k, advanced = n, True
                        break
            if not advanced:
                return None
        return links

    def _features(self, f):
        if f not in self._sift:
            g, m, _ = self.frames[f]
            self._sift[f] = self._sift_detector.detectAndCompute(g, m)
        return self._sift[f]

    def _bridge(self, a, b, min_inliers=25):
        key = ('sift', a, b)
        if key not in self._links:
            (ka, da), (kb, db) = self._features(a), self._features(b)
            result, pairs = (None, 0), None
            if da is not None and db is not None and len(ka) >= min_inliers and len(kb) >= min_inliers:
                matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
                good = [m for m, n2 in (x for x in matches if len(x) == 2) if m.distance < 0.75 * n2.distance]
                if len(good) >= min_inliers:
                    P0 = np.float32([ka[m.queryIdx].pt for m in good])
                    P1 = np.float32([kb[m.trainIdx].pt for m in good])
                    H, inl = cv2.findHomography(P0, P1, cv2.RANSAC, 3.0)
                    n = int(inl.sum()) if inl is not None else 0
                    if H is not None and n >= min_inliers:
                        keep = inl.ravel().astype(bool)
                        result, pairs = (H, n), (P0[keep], P1[keep])
            self._links[key] = result
            self._pairs[(a, b)] = pairs if pairs is not None else self._pairs.get((a, b))
            self.bridges_tried += 1
            self.bridges_ok += result[0] is not None
        return self._links[key]

    def _to_full(self, H_small, s_src, s_dst):
        return np.diag([1 / s_dst, 1 / s_dst, 1]) @ H_small @ np.diag([s_src, s_src, 1])

    # ------------------------------------------------------------------ public API
    def register(self, src, dst):
        if src == dst:
            return Registration(True, np.eye(3), 0, 10 ** 6, 0)
        keys = sorted(self.frames)
        i, j = keys.index(src), keys.index(dst)
        step = 1 if j > i else -1
        H, links, weakest, k = np.eye(3), 0, 10 ** 6, i
        lo, hi = min(i, j), max(i, j)
        while k != j:
            # Try the next frame; skip up to two frames past a failed link (e.g. motion blur).
            advanced = False
            for hop in (1, 2, 3):
                n = k + step * hop
                if not lo <= n <= hi:
                    break
                Hl, inl = self._link(keys[k], keys[n])
                if Hl is not None:
                    H, links, weakest, k, advanced = Hl @ H, links + 1, min(weakest, inl), n, True
                    break
            if not advanced:
                return Registration(False, np.eye(3), links, 0, 0)

        # Drift removal: re-track between the warped source and the destination.
        gs, ms, ss = self.frames[src]
        gd, _, sd = self.frames[dst]
        warped = cv2.warpPerspective(gs, H, (gd.shape[1], gd.shape[0]))
        wmask = cv2.warpPerspective(ms, H, (gd.shape[1], gd.shape[0]), flags=cv2.INTER_NEAREST)
        wmask = cv2.erode(wmask, np.ones((15, 15), np.uint8))  # avoid warped image borders
        Hc, refine = self._track(warped, wmask, gd)
        if Hc is not None and refine >= 2 * self.min_inliers:
            H = Hc @ H
        else:
            refine = 0
        return Registration(True, self._to_full(H, ss, sd), links, weakest, refine)


    # ------------------------------------------------------------------ point mapping (scene canvases)
    def track_points(self, src, dst, points, corridor_frac=0.2, min_local=10):
        """Carry full-resolution points from frame src to frame dst link by link.

        The points are treated as one measured object (e.g. a measurement's two endpoints). At every
        link they move with a RANSAC homography fitted ONLY to tracked features inside a corridor
        around them (corridor_frac of the frame width around the segment/polyline). Under camera
        translation a whole-frame homography follows whichever surface has the most texture (floor,
        far wall); the corridor keeps the fit on or near the measured surface. With too few corridor
        features the link's whole-frame homography is used and counted as a 'global' move.
        Returns (points, stats) or (None, stats) if the frames cannot be chained."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if src == dst:
            return pts.copy(), {'links': 0, 'local_moves': 0, 'global_moves': 0, 'min_inliers': 10 ** 6,
                                'H_small': np.eye(3)}
        chain = self._chain(src, dst)
        if chain is None:
            return None, {'links': 0, 'local_moves': 0, 'global_moves': 0}
        _, _, s = self.frames[src]
        q = pts * s
        R = corridor_frac * self.work_width
        local = glob = 0
        H_total = np.eye(3)
        for a, b, H, _ in chain:
            P0, P1 = self._pairs.get((a, b)) or (None, None)
            Hc = None
            if P0 is not None:
                near = _near_polyline(P0, q, R)
                if near.sum() >= min_local:
                    Hc, inl = cv2.findHomography(P0[near], P1[near], cv2.RANSAC, 2.0)
                    if Hc is None or inl is None or inl.sum() < min_local:
                        Hc = None
            if Hc is not None:
                local += 1
            else:
                Hc = H
                glob += 1
            q = cv2.perspectiveTransform(q.reshape(-1, 1, 2), Hc).reshape(-1, 2)
            H_total = Hc @ H_total
        _, _, sd = self.frames[dst]
        return q / sd, {'links': len(chain), 'local_moves': local, 'global_moves': glob,
                        'min_inliers': int(min(c[3] for c in chain)), 'H_small': H_total}

    def _patch_ncc(self, src, dst, point_full, H_small, radius_frac=0.12):
        """Appearance check for a mapped point: sample a patch of SCENE pixels around it in the source
        (tape, housing and skin are masked out), map the samples with the composed transform and
        correlate with the destination. A transform that is self-consistent (passes the round trip)
        but follows the wrong depth layer - e.g. a far beach instead of a near railing - puts the
        point on different surfaces in the two frames and fails here. None if too few scene pixels."""
        gs, ms, s = self.frames[src]
        gd, _, sd = self.frames[dst]
        c = np.asarray(point_full, dtype=np.float64) * s
        r = radius_frac * self.work_width
        ys, xs = np.mgrid[-r:r:40j, -r:r:40j]
        P = np.stack([xs.ravel() + c[0], ys.ravel() + c[1]], 1)
        inside = (P[:, 0] >= 0) & (P[:, 0] < gs.shape[1] - 1) & (P[:, 1] >= 0) & (P[:, 1] < gs.shape[0] - 1)
        P = P[inside]
        if len(P) < 100:
            return None
        valid = ms[P[:, 1].astype(int), P[:, 0].astype(int)] > 0
        P = P[valid]
        if len(P) < 100:
            return None
        Q = cv2.perspectiveTransform(P.reshape(-1, 1, 2), H_small).reshape(-1, 2) * (sd / s)
        okq = (Q[:, 0] >= 0) & (Q[:, 0] < gd.shape[1] - 1) & (Q[:, 1] >= 0) & (Q[:, 1] < gd.shape[0] - 1)
        if okq.sum() < 100:
            return None
        sample = lambda g, X: cv2.remap(g, X[:, 0].astype(np.float32).reshape(-1, 1),
                                        X[:, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).ravel()
        return _ncc(sample(gs, P[okq]).astype(np.float64), sample(gd, Q[okq]).astype(np.float64))

    def map_validated(self, src, dst, points, tol_frac=0.02):
        """track_points src->dst, then back dst->src through independently computed reverse links.
        A point is reliable only if it returns within tol_frac of the frame diagonal (cycle check)
        and lands inside the destination frame. Returns (points, ok mask, cycle errors px, stats)."""
        fwd, st = self.track_points(src, dst, points)
        n = len(np.asarray(points).reshape(-1, 2))
        if fwd is None:
            return None, np.zeros(n, bool), np.full(n, np.inf), st
        back, st_b = self.track_points(dst, src, fwd)
        if back is None:
            return fwd, np.zeros(n, bool), np.full(n, np.inf), st
        g, _, s = self.frames[dst]
        Hh, Ww = g.shape[0] / s, g.shape[1] / s
        err = np.linalg.norm(back - np.asarray(points, dtype=np.float64).reshape(-1, 2), axis=1)
        inside = (fwd[:, 0] >= 0) & (fwd[:, 0] < Ww) & (fwd[:, 1] >= 0) & (fwd[:, 1] < Hh)
        cycle_ok = err <= tol_frac * np.hypot(Ww, Hh)
        ok = cycle_ok & inside
        ncc = [self._patch_ncc(src, dst, p, st['H_small']) if o else None
               for p, o in zip(np.asarray(points, dtype=np.float64).reshape(-1, 2), ok)]
        # Every endpoint with enough visible scene around it must match (>= min_patch_ncc); at least
        # one endpoint must be checkable. Measured on the footage: correct mappings 0.97-1.0, mappings
        # that followed the wrong depth layer 0.20-0.31. A housing end is usually surrounded by
        # housing/hand pixels only, so it is often 'inconclusive' (None) rather than failed.
        appearance_ok = all(c is None or c >= self.min_patch_ncc for c in ncc) and \
            any(c is not None for c in ncc)
        ok &= appearance_ok
        st.update(cycle_error_px=[round(float(e), 1) for e in err], back_links=st_b.get('links'),
                  inside=[bool(x) for x in inside], cycle_ok=[bool(x) for x in cycle_ok], patch_ncc=[None if c is None else round(c, 2) for c in ncc])
        st.pop('H_small', None)
        return fwd, ok, err, st


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-6 else -1.0


def _near_polyline(P, poly, R):
    """Mask of points P within distance R of the polyline through `poly` (or of a single point)."""
    if len(poly) == 1:
        return np.linalg.norm(P - poly[0], axis=1) < R
    best = np.full(len(P), np.inf)
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-9), 0, 1)
        best = np.minimum(best, np.linalg.norm(P - (a + t[:, None] * ab), axis=1))
    return best < R


def map_points(H, points):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)
