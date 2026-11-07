"""Run the measurement pipeline on one video or every video in a folder.

    .venv/bin/python -m reekon data/inbox                       # classical CPU strategy
    python3 -m reekon data/inbox --detector hand-assisted       # + GPU hand pose (system python3)
    .venv/bin/python -m reekon data/inbox/Cabinet-1.MOV --out outputs
"""
import argparse
import json
from pathlib import Path

from .pipeline import run

VIDEO_EXTS = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input', type=Path, help='video file or folder of videos')
    p.add_argument('--out', type=Path, default=Path('outputs'))
    p.add_argument('--detector', choices=['classical', 'hand-assisted'], default='classical')
    p.add_argument('--sample-fps', type=float, default=10.0)
    p.add_argument('--no-proxy', action='store_true', help='skip the browser H.264 copy of the video')
    args = p.parse_args()

    videos = [args.input] if args.input.is_file() else sorted(
        f for f in args.input.iterdir() if f.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise SystemExit(f'no videos found in {args.input}')
    index = []
    for v in videos:
        out = args.out / v.stem / args.detector
        r = run(v, out, args.sample_fps, args.detector, proxy=not args.no_proxy)
        index.append({'video': v.name, 'detector': args.detector,
                      'results': str((out / 'results.json').relative_to(args.out)),
                      'measurements': r['counts']['measurements']})
    # Merge into the run index the viewer reads.
    idx_path = args.out / 'index.json'
    old = json.loads(idx_path.read_text()) if idx_path.exists() else []
    keep = [e for e in old if (e['video'], e['detector']) not in {(i['video'], i['detector']) for i in index}]
    idx_path.write_text(json.dumps(sorted(keep + index, key=lambda e: (e['video'], e['detector'])), indent=1))


if __name__ == '__main__':
    main()
