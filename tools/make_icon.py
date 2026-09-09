"""Dev tool: build assets/AppIcon.icns, with the app writing its own icon."""
import os
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hw import render  # noqa: E402
from hw.engine import Model  # noqa: E402

SIZE = 1024
INK = '#141414'
PAPER = '#fbfbf8'
SIZES = [16, 32, 128, 256, 512]


def glyph_image():
    strokes = Model().generate(['Aa'], style=9, bias=1.2, seed=11)
    polylines, page = render.layout(strokes, margin=0)

    tile = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(tile).rounded_rectangle([0, 0, SIZE - 1, SIZE - 1], radius=int(SIZE * 0.225),
                                           fill=PAPER)
    inner = SIZE * 0.52
    scale = min(inner / page[0], inner / page[1])
    # to_image draws the pen at stroke_width * dpi_scale pixels
    ink = render.to_image(polylines, page, stroke_width=12.0 / scale, color=INK,
                          background=None, dpi_scale=scale, supersample=3)
    tile.alpha_composite(ink, ((SIZE - ink.width) // 2, (SIZE - ink.height) // 2))
    return tile


def main():
    icon = glyph_image()
    assets = os.path.join(ROOT, 'assets')
    os.makedirs(assets, exist_ok=True)
    icon.save(os.path.join(assets, 'icon.png'))

    iconset = os.path.join(assets, 'AppIcon.iconset')
    shutil.rmtree(iconset, ignore_errors=True)
    os.makedirs(iconset)
    for size in SIZES:
        icon.resize((size, size), Image.LANCZOS).save(
            os.path.join(iconset, 'icon_{0}x{0}.png'.format(size)))
        icon.resize((size * 2, size * 2), Image.LANCZOS).save(
            os.path.join(iconset, 'icon_{0}x{0}@2x.png'.format(size)))

    icns = os.path.join(assets, 'AppIcon.icns')
    subprocess.run(['iconutil', '-c', 'icns', iconset, '-o', icns], check=True)
    shutil.rmtree(iconset)
    print('wrote', icns, os.path.getsize(icns), 'bytes')


if __name__ == '__main__':
    main()
