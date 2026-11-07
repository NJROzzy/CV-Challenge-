"""Inspect every video in a drop folder before building the measurement pipeline.

For each video in --inbox this writes to <outbox>/<video name>/:
  metadata.json      fps, frame count, resolution, duration
  contact_sheet.jpg  evenly spaced sampled frames with timestamps (and hand keypoints)
  hands.json         per sampled frame: hand keypoints from TensorRT Pose-ResNet18-Hand

This is an exploration tool: it does NOT detect measurements or tape endpoints.

Run with the system Python (it has jetson_inference + OpenCV):
    /usr/bin/python3 tools/inspect_videos.py
    /usr/bin/python3 tools/inspect_videos.py --no-hands      # CPU-only, no model
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm'}
ROOT = Path(__file__).resolve().parent.parent


def sample_frames(path, sample_fps):
    """Decode sequentially; keep frames at roughly sample_fps. Returns metadata, [(index, t, bgr)]."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'cannot open {path}')
    # Phone footage is often variable-frame-rate: the header rate (e.g. 120) can be far from
    # the real rate (~30), so timestamps come from each decoded frame, not index / header fps.
    meta = {'file': path.name, 'header_fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count_reported': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    frames, index, next_t, t = [], 0, 0.0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = max(0.0, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        if t >= next_t:
            frames.append((index, t, frame))
            next_t += 1 / sample_fps
        index += 1
    cap.release()
    duration = t + (t / max(index - 1, 1))  # last timestamp plus one average frame interval
    meta.update(frames_decoded=index, duration_s=duration, fps=index / duration if duration else None,
                samples=len(frames))
    if frames:  # decoded size after any rotation metadata is applied
        meta['decoded_height'], meta['decoded_width'] = frames[0][2].shape[:2]
    return meta, frames


class HandPose:
    def __init__(self, threshold, data_dir):
        import os
        from jetson_inference import poseNet
        from jetson_utils import cudaFromNumpy
        # jetson-inference resolves 'networks/...' model paths relative to the working directory.
        cwd = os.getcwd()
        os.chdir(data_dir)
        try:
            self.net = poseNet('resnet18-hand', threshold=threshold)
        finally:
            os.chdir(cwd)
        self.to_cuda = cudaFromNumpy

    def __call__(self, bgr):
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        poses = self.net.Process(self.to_cuda(rgb), overlay='none')
        return [[{'name': self.net.GetKeypointName(k.ID), 'x': float(k.x), 'y': float(k.y)}
                 for k in p.Keypoints] for p in poses]


def draw_hands(img, hands, scale):
    for hand in hands:
        for k in hand:
            cv2.circle(img, (int(k['x'] * scale), int(k['y'] * scale)), 3, (0, 255, 0), -1)


def contact_sheet(frames, hands_by_index, tiles, tile_width, cols):
    picks = np.linspace(0, len(frames) - 1, min(tiles, len(frames))).round().astype(int)
    images = []
    for i in picks:
        index, t, bgr = frames[i]
        scale = tile_width / bgr.shape[1]
        tile = cv2.resize(bgr, (tile_width, round(bgr.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        draw_hands(tile, hands_by_index.get(index, []), scale)
        label = f'#{index}  t={t:.1f}s'
        cv2.putText(tile, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        images.append(tile)
    h = max(im.shape[0] for im in images)
    images = [cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT) for im in images]
    images += [np.zeros_like(images[0])] * (-len(images) % cols)
    return np.vstack([np.hstack(images[r:r + cols]) for r in range(0, len(images), cols)])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--inbox', type=Path, default=ROOT / 'data' / 'inbox')
    p.add_argument('--outbox', type=Path, default=ROOT / 'data' / 'outbox')
    p.add_argument('--sample-fps', type=float, default=2.0)
    p.add_argument('--tiles', type=int, default=24)
    p.add_argument('--cols', type=int, default=6)
    p.add_argument('--tile-width', type=int, default=320)
    p.add_argument('--no-hands', action='store_true', help='skip the TensorRT hand-pose model')
    p.add_argument('--hand-threshold', type=float, default=0.15)
    p.add_argument('--models-dir', type=Path, default=Path.home() / 'jetson-inference' / 'data',
                   help='directory containing jetson-inference networks/')
    args = p.parse_args()

    videos = sorted(f for f in args.inbox.glob('*') if f.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print(f'No videos in {args.inbox} (extensions: {", ".join(sorted(VIDEO_EXTS))})')
        return
    hand_pose = None if args.no_hands else HandPose(args.hand_threshold, args.models_dir)

    summary = []
    for video in videos:
        out = args.outbox / video.stem
        out.mkdir(parents=True, exist_ok=True)
        meta, frames = sample_frames(video, args.sample_fps)
        hands = {}
        if hand_pose:
            hands = {index: hand_pose(bgr) for index, _, bgr in frames}
            (out / 'hands.json').write_text(json.dumps(
                [{'frame': i, 't': t, 'hands': hands[i]} for i, t, _ in frames], indent=1))
            meta['sampled_frames_with_hands'] = sum(bool(h) for h in hands.values())
        (out / 'metadata.json').write_text(json.dumps(meta, indent=2))
        if frames:
            cv2.imwrite(str(out / 'contact_sheet.jpg'),
                        contact_sheet(frames, hands, args.tiles, args.tile_width, args.cols))
        summary.append(meta)
        hand_note = f", hands in {meta['sampled_frames_with_hands']}/{len(frames)} samples" if hand_pose else ''
        print(f"{video.name}: {meta['decoded_width']}x{meta['decoded_height']} @ {meta['fps']:.1f} fps, "
              f"{meta['duration_s']:.1f}s, {len(frames)} samples{hand_note} -> {out}")
    (args.outbox / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
