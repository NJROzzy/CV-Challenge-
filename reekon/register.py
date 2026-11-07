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
        return (H, n) if H is not None and n >= self.min_inliers else (None, n)

    def _link(self, a, b):
        key = (a, b)
        if key not in self._links:
            ga, ma, _ = self.frames[a]
            gb, _, _ = self.frames[b]
            self._links[key] = self._track(ga, ma, gb)
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


def map_points(H, points):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)
