"""Check a running server: python tests/smoke_api.py --url http://localhost:8000."""
import argparse
import json
from io import BytesIO
from pathlib import Path
import struct
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from PIL import Image


def check_server(base_url):
    base_url = base_url.rstrip('/')
    for attempt in range(60):
        try:
            with urlopen(base_url + '/', timeout=5) as response:
                loaded = json.load(response)['models']
            break
        except (URLError, TimeoutError):
            if attempt == 59:
                raise
            time.sleep(0.5)
    assert {'ebm', 'diffusion', 'gan'}.issubset(loaded), loaded
    results = {'models': loaded, 'requests': []}
    for name, expected in [('ebm', 32), ('diffusion', 32), ('gan', 28)]:
        url = f'{base_url}/generate/{name}?num_samples=1&seed=42'
        started = time.monotonic()
        with urlopen(url, timeout=120) as response:
            assert response.status == 200
            assert response.headers.get_content_type() == 'image/png'
            data = response.read()
        elapsed = time.monotonic() - started
        assert data[:8] == b'\x89PNG\r\n\x1a\n'
        assert data[12:16] == b'IHDR'
        dimensions = struct.unpack('>II', data[16:24])
        assert dimensions == (expected, expected), dimensions
        Image.open(BytesIO(data)).verify()
        with urlopen(url, timeout=120) as response:
            assert response.read() == data, f'{name}: seed did not reproduce the image'
        results['requests'].append(dict(model=name, status=200,
                                        width=expected, height=expected,
                                        seconds=round(elapsed, 3), repeatable=True))
    for path, expected in [('/generate/unknown', 404),
                           ('/generate/ebm?num_samples=0', 422),
                           ('/generate/diffusion?seed=-1', 422)]:
        try:
            urlopen(base_url + path, timeout=10)
        except HTTPError as error:
            assert error.code == expected, (path, error.code)
        else:
            raise AssertionError(f'{path}: expected HTTP {expected}')
    results['error_responses'] = '404 and 422 checked'
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://localhost:8000')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = json.dumps(check_server(args.url), indent=2) + '\n'
    if args.report:
        args.report.write_text(report)
    print(report, end='')
