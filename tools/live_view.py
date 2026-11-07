"""Pick a video and watch it play with the GPU hand-pose model drawn on top.

    python3 tools/live_view.py                 # choose from data/inbox
    python3 tools/live_view.py path/to/video.MOV
    python3 tools/live_view.py --save out.mp4  # no window: write the annotated video instead

Keys in the window:  space = pause/resume,  n = next frame (while paused),  q / Esc = quit

This only shows HAND keypoints from NVIDIA's pretrained Pose-ResNet18-Hand model.
It does not detect the tape, measurements or endpoints yet.
Use the system python3 (it has jetson_inference); the window opens on the Jetson's screen.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
VIDEO_EXTS = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm'}
WINDOW = 'REEKON live view (space pause, n step, q quit)'


def choose_video(inbox):
    videos = sorted(f for f in inbox.glob('*') if f.suffix.lower() in VIDEO_EXTS)
    if not videos:
        sys.exit(f'No videos in {inbox}. Copy some there or pass a path.')
    for i, v in enumerate(videos, 1):
        print(f'  {i}. {v.name}  ({v.stat().st_size / 2**20:.0f} MB)')
    while True:
        answer = input(f'Pick a video [1-{len(videos)}]: ').strip()
        if answer.isdigit() and 1 <= int(answer) <= len(videos):
            return videos[int(answer) - 1]
        print('Please type one of the numbers above.')


def load_hand_model(models_dir, threshold):
    from jetson_inference import poseNet
    cwd = os.getcwd()
    os.chdir(models_dir)  # jetson-inference resolves 'networks/...' relative to the working directory
    try:
        return poseNet('resnet18-hand', threshold=threshold)
    finally:
        os.chdir(cwd)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('video', nargs='?', type=Path, help='video file (default: choose from data/inbox)')
    p.add_argument('--inbox', type=Path, default=ROOT / 'data' / 'inbox')
    p.add_argument('--height', type=int, default=900, help='display/processing height in pixels')
    p.add_argument('--save', type=Path, help='write annotated video here instead of opening a window')
    p.add_argument('--no-hands', action='store_true', help='just play the video, no model')
    p.add_argument('--hand-threshold', type=float, default=0.15)
    p.add_argument('--models-dir', type=Path, default=Path.home() / 'jetson-inference' / 'data')
    args = p.parse_args()

    video = args.video or choose_video(args.inbox)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f'Cannot open {video}')

    if not args.save and not os.environ.get('DISPLAY'):
        os.environ['DISPLAY'] = ':0'  # e.g. running from a VS Code / SSH terminal: use the Jetson's screen
        print('DISPLAY not set; opening the window on the Jetson screen (:0).')

    net = None
    if not args.no_hands:
        from jetson_utils import cudaFromNumpy, cudaToNumpy, cudaDeviceSynchronize
        net = load_hand_model(args.models_dir, args.hand_threshold)

    writer, paused, step, index, last_t = None, False, False, 0, None
    print(f'Playing {video.name}. Press q in the window to quit.')
    while True:
        if not paused or step:
            ok, frame = cap.read()
            if not ok:
                break
            step = False
            t = max(0.0, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            scale = args.height / frame.shape[0]
            frame = cv2.resize(frame, (round(frame.shape[1] * scale), args.height), interpolation=cv2.INTER_AREA)

            started = time.perf_counter()
            hands = 0
            if net:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cudaFromNumpy(rgb)
                hands = len(net.Process(img, overlay='links,keypoints'))
                cudaDeviceSynchronize()
                frame = cv2.cvtColor(cudaToNumpy(img), cv2.COLOR_RGB2BGR)
            model_ms = (time.perf_counter() - started) * 1000

            lines = [f'{video.name}  frame {index}  t={t:.1f}s']
            if net:  # one real hand is often split into several partial detections
                lines.append(f'hand detections: {hands}   model {model_ms:.0f} ms')
            for row, text in enumerate(lines):
                y = 28 + 28 * row
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
            index += 1

            if args.save:
                if writer is None:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    fps = 30.0 if not fps or fps > 60 else fps  # header rate can be wrong on phone video
                    args.save.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                             (frame.shape[1], frame.shape[0]))
                writer.write(frame)
                if index % 30 == 0:
                    print(f'\r  {index} frames written', end='', flush=True)
                continue

            # Pace playback to the video's own timestamps.
            if last_t is not None:
                wait = (t - last_t) - (time.perf_counter() - last_wall)
                if wait > 0:
                    time.sleep(wait)
            last_t, last_wall = t, time.perf_counter()
            cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord(' '):
            paused, last_t = not paused, None
        elif key == ord('n') and paused:
            step = True
        if not args.save and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break  # window closed with the mouse

    cap.release()
    if writer:
        writer.release()
        print(f'\nSaved {args.save}')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
