"""Stage 5: annotated output images.

Visual language (also used by the browser viewer):
  * numbered line from endpoint A (hook) to endpoint B (housing / blade entry point);
  * filled marker  = endpoint seen directly in this frame;
  * hollow marker + dashed ring = endpoint mapped in from other frames by registration
    (ring radius = observed spread, i.e. positional uncertainty);
  * a missing endpoint is written in the caption and never drawn;
  * when an endpoint lies outside the reference frame, the frame it was seen in is warped
    around the reference frame (registered mosaic, dimmed) and the reference frame is outlined.
"""
import math

import cv2
import numpy as np

COLORS = [(255, 64, 255), (0, 200, 255), (60, 220, 60), (255, 160, 0), (80, 80, 255), (200, 200, 0)]


def color_for(measurement_id):
    return COLORS[(measurement_id - 1) % len(COLORS)]


def _dashed_circle(img, center, radius, color, thickness, dashes=24):
    for k in range(0, dashes, 2):
        a0, a1 = 360 * k / dashes, 360 * (k + 1) / dashes
        cv2.ellipse(img, center, (radius, radius), 0, a0, a1, color, thickness, cv2.LINE_AA)


def _label(img, text, org, scale, color, thickness):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness * 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def build_canvas(canvas_img, context_img=None, H_context=None, max_scale=2.5):
    """Return (image, T) where T maps canvas-frame pixels into the returned image."""
    h, w = canvas_img.shape[:2]
    if context_img is None:
        return canvas_img.copy(), np.eye(3), None
    ch, cw = context_img.shape[:2]
    corners = cv2.perspectiveTransform(np.float32([[[0, 0], [cw, 0], [cw, ch], [0, ch]]]), H_context)[0]
    allp = np.vstack([corners, [[0, 0], [w, h]]])
    x0, y0 = np.floor(allp.min(0))
    x1, y1 = np.ceil(allp.max(0))
    # Bound the mosaic so a degenerate homography cannot explode the image.
    x0, y0 = max(x0, -w * (max_scale - 1) / 2), max(y0, -h * (max_scale - 1) / 2)
    x1, y1 = min(x1, w + w * (max_scale - 1) / 2), min(y1, h + h * (max_scale - 1) / 2)
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
    size = (int(x1 - x0), int(y1 - y0))
    out = cv2.warpPerspective(context_img, T @ H_context, size, borderValue=(40, 40, 40))
    out = (out * 0.65 + 40 * 0.35).astype(np.uint8)  # dim the registered context
    ox, oy = int(-x0), int(-y0)
    out[oy:oy + h, ox:ox + w] = canvas_img
    outline = (ox, oy, ox + w - 1, oy + h - 1)
    return out, T, outline


def draw_measurement(img, m, T, scale_ref):
    """Draw one measurement (dict form) onto img. Coordinates in m are canvas-frame pixels."""
    color = color_for(m['id'])
    th = max(2, int(round(scale_ref / 260)))
    r = max(8, int(round(scale_ref / 90)))
    pts = {}
    for key in ('hook', 'housing'):
        e = m[key]
        if e['status'] == 'missing':
            continue
        p = cv2.perspectiveTransform(np.float64([[[e['x'], e['y']]]]), T)[0, 0]
        pts[key] = (int(round(p[0])), int(round(p[1])))
    if len(pts) == 2:
        cv2.line(img, pts['hook'], pts['housing'], (0, 0, 0), th * 3, cv2.LINE_AA)
        cv2.line(img, pts['hook'], pts['housing'], color, th, cv2.LINE_AA)
    for key, letter in (('hook', 'A'), ('housing', 'B')):
        if key not in pts:
            continue
        e, p = m[key], pts[key]
        if e['status'] == 'observed':
            if key == 'hook':
                cv2.circle(img, p, r, color, -1, cv2.LINE_AA)
            else:
                cv2.rectangle(img, (p[0] - r, p[1] - r), (p[0] + r, p[1] + r), color, -1)
            cv2.circle(img, p, r + th, (0, 0, 0), th, cv2.LINE_AA)
        else:
            if key == 'hook':
                cv2.circle(img, p, r, color, th + 1, cv2.LINE_AA)
            else:
                cv2.rectangle(img, (p[0] - r, p[1] - r), (p[0] + r, p[1] + r), color, th + 1)
            ring = int(max(2 * r, e.get('spread_px') or 0))
            _dashed_circle(img, p, ring, color, max(1, th - 1))
        _label(img, f"{m['id']}{letter}", (p[0] + r + 6, p[1] - r - 6), scale_ref / 900, color, th)
    if len(pts) == 2:
        mid = ((pts['hook'][0] + pts['housing'][0]) // 2, (pts['hook'][1] + pts['housing'][1]) // 2)
        _label(img, f"#{m['id']}", (mid[0] + r, mid[1] - r), scale_ref / 700, color, th + 1)
    return pts


def caption(img, lines):
    """Return img with a caption band added ABOVE it (never covering an endpoint)."""
    width = img.shape[1]
    pad = max(8, width // 80)
    th = max(1, width // 900)
    tw = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 1.0, th)[0][0] for t in lines)
    s = min(width / 1100, (width - 2 * pad) / tw)
    line_h = int(42 * s) + pad // 2
    band = np.full((pad + line_h * len(lines), width, 3), 25, np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(band, text, (pad, pad + line_h * i + int(30 * s)), cv2.FONT_HERSHEY_SIMPLEX, s,
                    (255, 255, 255), th, cv2.LINE_AA)
    return np.vstack([band, img])


def render(m, canvas_img, context_img=None, H_context=None, video_name=''):
    img, T, outline = build_canvas(canvas_img, context_img, H_context)
    scale_ref = math.hypot(*canvas_img.shape[:2])
    if outline is not None:
        th = max(2, int(scale_ref / 400))
        cv2.rectangle(img, outline[:2], outline[2:], (255, 255, 255), th)
        _label(img, 'reference frame', (outline[0] + 10, outline[3] - 15), scale_ref / 1400, (255, 255, 255), 2)
    draw_measurement(img, m, T, scale_ref)
    status = lambda e: e['status'] if e['status'] == 'missing' else f"{e['status']} (conf {e['confidence']:.2f}, {e['support']}/{e['candidates']} frames agree)"
    lines = [f"{video_name}  measurement #{m['id']}  {m['start_t']:.1f}-{m['end_t']:.1f}s  reference t={m['canvas_t']:.2f}s",
             f"A hook: {status(m['hook'])}",
             f"B housing: {status(m['housing'])}"]
    if outline is not None:
        shift = m.get('context_shift_px')
        extra = f', shifted {shift:.0f}px by the straight-tape correction' if shift else ''
        lines.append(f"dimmed area: frame t={m.get('context_t', 0):.2f}s registered into this view (approximate{extra})")
    lines.append('filled marker = seen in this frame; hollow + dashed ring = mapped from other frames')
    return caption(img, lines)
