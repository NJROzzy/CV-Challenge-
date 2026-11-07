"""Run the measurement pipeline on one video or every video in a folder.

    .venv/bin/python -m reekon data/inbox                       # classical CPU strategy
    python3 -m reekon data/inbox --detector hand-assisted       # + GPU hand pose (system python3)
    .venv/bin/python -m reekon data/inbox/Cabinet-1.MOV --out outputs
"""
import argparse
import json
from pathlib import Path

from .config import SAMPLE_FPS, VIDEO_EXTS, DETECTORS
from .pipeline import run
from .video import validate_sample_fps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input', type=Path, help='video file or folder of videos')
    p.add_argument('--out', type=Path, default=Path('outputs'))
    p.add_argument('--detector', choices=DETECTORS, default='classical')
    p.add_argument('--sample-fps', type=float, default=SAMPLE_FPS)
    p.add_argument('--no-proxy', action='store_true', help='skip the browser H.264 copy of the video')
    args = p.parse_args()
    try:
        validate_sample_fps(args.sample_fps)
    except ValueError as exc:
        p.error(str(exc))
    if not args.input.exists():
        p.error(f'input does not exist: {args.input}')

    videos = [args.input] if args.input.is_file() else sorted(
        f for f in args.input.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS)
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
    updated = {(entry['video'], entry['detector']) for entry in index}
    keep = [e for e in old if (e['video'], e['detector']) not in updated]
    idx_path.write_text(json.dumps(sorted(keep + index, key=lambda e: (e['video'], e['detector'])), indent=1))


if __name__ == '__main__':
    main()
