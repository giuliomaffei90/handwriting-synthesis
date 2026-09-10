"""Dev tool: pack the model, the styles and the test trace for the Swift app.

Swift has no numpy, so everything it needs is written into one simple
container per file:

    "HWB1" | uint32 count | entries...
    entry: uint32 nameLen | name | uint8 type | uint32 ndim | uint32 dims[ndim] | payload
    type 0 = float32 little-endian, row major;  type 1 = utf8 bytes

Run: .venv/bin/python tools/make_swift_resources.py
"""
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'swift', 'Resources')
sys.path.insert(0, ROOT)

from hw import engine  # noqa: E402


def write(path, entries):
    """entries: list of (name, numpy array) or (name, str)."""
    with open(path, 'wb') as f:
        f.write(b'HWB1')
        f.write(struct.pack('<I', len(entries)))
        for name, value in entries:
            f.write(struct.pack('<I', len(name)))
            f.write(name.encode('utf-8'))
            if isinstance(value, str):
                payload = value.encode('utf-8')
                f.write(struct.pack('<BI', 1, 1))
                f.write(struct.pack('<I', len(payload)))
                f.write(payload)
            else:
                array = np.ascontiguousarray(value, dtype='<f4')
                f.write(struct.pack('<BI', 0, array.ndim))
                for dim in array.shape:
                    f.write(struct.pack('<I', dim))
                f.write(array.tobytes())
    return os.path.getsize(path)


def main():
    os.makedirs(OUT, exist_ok=True)

    with np.load(engine.WEIGHTS_PATH) as npz:
        weights = [(k, npz[k]) for k in npz.files if k != 'alphabet_size']
    size = write(os.path.join(OUT, 'model.bin'), weights)
    print('model.bin   {:>5.1f} MB  {} tensors'.format(size / 1e6, len(weights)))

    previews_path = os.path.join(ROOT, 'hw', 'previews.npz')
    previews = np.load(previews_path) if os.path.exists(previews_path) else {}

    styles = []
    for s in engine.available_styles():
        styles.append(('style{}.name'.format(s), engine.style_name(s)))
        styles.append(('style{}.text'.format(s), engine.style_text(s)))
        styles.append(('style{}.strokes'.format(s), engine.style_strokes(s)))
        key = 'style{}'.format(s)
        if key in previews:
            styles.append((key + '.preview', previews[key]))
    size = write(os.path.join(OUT, 'styles.bin'), styles)
    print('styles.bin  {:>5.1f} MB  {} styles'.format(
        size / 1e6, len(engine.available_styles())))

    reference = os.path.join(ROOT, 'tests', 'reference.npz')
    with np.load(reference) as npz:
        entries = [(k, npz[k]) for k in ('x', 'x_len', 'c', 'c_len', 'params')]
        entries += [('state_' + k, npz['state_' + k]) for k in engine.STATE_FIELDS]
    size = write(os.path.join(OUT, 'reference.bin'), entries)
    print('reference.bin {:>3.1f} MB  the TensorFlow trace the Swift port is checked against'
          .format(size / 1e6))


if __name__ == '__main__':
    main()
