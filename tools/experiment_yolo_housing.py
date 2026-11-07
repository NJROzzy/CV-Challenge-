"""Experiment: can a pretrained COCO detector find the tape-measure housing?

No model is trained. YOLOv8n (COCO, 80 classes) has no "tape measure" class, but this tape measure's
housing - a dark handheld body with an LCD - may be detected as a phone-like object. For every
sampled frame we compare YOLO's boxes with the classical detector's housing end:

  * agreement: frames where the classical detector SAW the housing and a YOLO box contains it;
  * YOLO-only: frames where YOLO fires on a candidate class but the classical detector found no tape
    (either a housing the classical detector missed, or a YOLO false positive - judged by eye from
    the saved grid, since there is no ground truth);
  * runtime per frame on this machine (CPU: the .venv PyTorch cannot use CUDA here).

    .venv/bin/python tools/experiment_yolo_housing.py data/inbox --weights data/outbox/models/yolov8n.pt
Writes data/outbox/experiments/yolo_housing.json and a review grid per video.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reekon.detect import ClassicalTapeDetector  # noqa: E402
from reekon.video import iter_frames  # noqa: E402

CANDIDATE_CLASSES = {'cell phone', 'remote', 'mouse', 'book', 'clock'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inbox', type=Path)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--sample-fps', type=float, default=1.0)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--out', type=Path, default=Path('data/outbox/experiments'))
    args = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    names = model.names
    det = ClassicalTapeDetector()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'weights': args.weights.name, 'conf': args.conf, 'sample_fps': args.sample_fps,
              'candidate_classes': sorted(CANDIDATE_CLASSES), 'videos': {}}
    for video in sorted(p for p in args.inbox.iterdir() if p.suffix.lower() in {'.mov', '.mp4'}):
        rows, tiles, yolo_ms, classical_ms = [], [], [], []
        for f in iter_frames(video, args.sample_fps):
            t0 = time.perf_counter()
            obs = det.detect(f.image, f.index, f.t)
            classical_ms.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            res = model.predict(f.image, imgsz=640, conf=args.conf, verbose=False)[0]
            yolo_ms.append((time.perf_counter() - t0) * 1000)
            boxes = [(names[int(c)], float(s), b.tolist()) for c, s, b in
                     zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxy)]
            cand = [b for b in boxes if b[0] in CANDIDATE_CLASSES]
            housing = obs.housing.xy if obs is not None and obs.housing_detected else None
            contains = housing is not None and any(
                b[2][0] - 20 <= housing[0] <= b[2][2] + 20 and b[2][1] - 20 <= housing[1] <= b[2][3] + 20 for b in cand)
            rows.append({'t': round(f.t, 2), 'classical_tape': obs is not None,
                         'classical_housing': housing is not None, 'yolo_candidates': len(cand),
                         'yolo_classes': sorted({b[0] for b in boxes}), 'yolo_box_contains_housing': contains})
            if cand or housing is not None:
                small = cv2.resize(f.image, (240, round(f.image.shape[0] * 240 / f.image.shape[1])))
                s = 240 / f.image.shape[1]
                for name, score, (x0, y0, x1, y1) in cand:
                    cv2.rectangle(small, (int(x0 * s), int(y0 * s)), (int(x1 * s), int(y1 * s)), (0, 255, 255), 2)
                    cv2.putText(small, f'{name} {score:.2f}', (int(x0 * s), max(12, int(y0 * s) - 3)), 0, .4, (0, 255, 255), 1)
                if housing is not None:
                    cv2.circle(small, (int(housing[0] * s), int(housing[1] * s)), 6, (0, 220, 0), -1)
                cv2.putText(small, f'{f.t:.1f}s', (3, 14), 0, .45, (0, 0, 255), 2)
                tiles.append(small)
        n = len(rows)
        with_h = [r for r in rows if r['classical_housing']]
        no_tape = [r for r in rows if not r['classical_tape']]
        summary = {
            'frames': n,
            'classical_housing_frames': len(with_h),
            'yolo_box_contains_classical_housing': sum(r['yolo_box_contains_housing'] for r in with_h),
            'frames_without_classical_tape': len(no_tape),
            'yolo_fires_without_classical_tape': sum(r['yolo_candidates'] > 0 for r in no_tape),
            'yolo_ms_per_frame_median': round(float(np.median(yolo_ms)), 1),
            'classical_ms_per_frame_median': round(float(np.median(classical_ms)), 1),
            'rows': rows}
        report['videos'][video.name] = summary
        print(video.name, {k: v for k, v in summary.items() if k != 'rows'})
        if tiles:
            h = max(t.shape[0] for t in tiles)
            tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles][:40]
            tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 8)
            cv2.imwrite(str(args.out / f'yolo_{video.stem}.jpg'),
                        np.vstack([np.hstack(tiles[r:r + 8]) for r in range(0, len(tiles), 8)]))
    (args.out / 'yolo_housing.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
