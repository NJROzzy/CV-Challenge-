"""Serve the results viewer on the local network (e.g. open it from a Mac on the same Wi-Fi).

    .venv/bin/python -m reekon.serve            # http://<jetson-ip>:8000
    .venv/bin/python -m reekon.serve --port 8080 --outputs outputs

Only three folders are exposed: viewer/ (the page), the outputs folder and results/ (benchmark
JSON). HTTP Range requests are supported because Safari will not play <video> without them.
"""
import argparse
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import re
import socket
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent


class Handler(SimpleHTTPRequestHandler):
    mounts = {}

    def translate_path(self, path):
        path = unquote(path.split('?', 1)[0].split('#', 1)[0])
        if path in ('', '/'):
            path = '/viewer/index.html'
        parts = [p for p in path.split('/') if p and p not in ('.', '..')]
        if not parts or parts[0] not in self.mounts:
            return str(ROOT / '__not_found__')
        return str(self.mounts[parts[0]].joinpath(*parts[1:]))

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Accept-Ranges', 'bytes')
        super().end_headers()

    def send_head(self):
        rng = self.headers.get('Range')
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        m = re.match(r'bytes=(\d*)-(\d*)$', rng.strip())
        size = os.path.getsize(path)
        if not m or (not m.group(1) and not m.group(2)):
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return None
        if m.group(1):
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else size - 1
        else:  # suffix range: last N bytes
            start, end = max(0, size - int(m.group(2))), size - 1
        if start >= size or end < start:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header('Content-Range', f'bytes */{size}')
            self.end_headers()
            return None
        end = min(end, size - 1)
        f = open(path, 'rb')
        f.seek(start)
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end - start + 1))
        self.end_headers()
        self._range_left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, '_range_left', None)
        if left is None:
            return super().copyfile(source, outputfile)
        while left > 0:
            chunk = source.read(min(64 * 1024, left))
            if not chunk:
                break
            outputfile.write(chunk)
            left -= len(chunk)
        self._range_left = None


def lan_addresses():
    addrs = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('192.0.2.1', 9))  # no packet is sent; picks the outbound interface
        addrs.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(addrs) or ['<jetson-ip>']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--host', default='0.0.0.0', help='0.0.0.0 = reachable from other machines on the LAN')
    p.add_argument('--outputs', type=Path, default=ROOT / 'outputs')
    args = p.parse_args()
    Handler.mounts = {'viewer': ROOT / 'viewer', 'outputs': args.outputs.resolve(), 'bench': ROOT / 'results'}
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler))
    print(f'Serving the REEKON viewer from {args.outputs}')
    for a in lan_addresses():
        print(f'  open  http://{a}:{args.port}/   (from your Mac on the same network)')
    print(f'  or    http://localhost:{args.port}/  (on the Jetson / via VS Code port forwarding)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
