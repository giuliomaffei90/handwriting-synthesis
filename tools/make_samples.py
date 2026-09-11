"""Dev tool: build the images used in readme.md."""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hw import render  # noqa: E402
from hw.engine import available_styles, style_name  # noqa: E402

DOCS = os.path.join(ROOT, 'docs')


def main():
    os.makedirs(DOCS, exist_ok=True)

    # the sheet shows exactly what the app's picker shows, so it reads the
    # preview lines rather than writing its own
    previews = np.load(os.path.join(ROOT, 'hw', 'previews.npz'))
    styles = available_styles()
    rows = [previews['style{}'.format(s)] for s in styles]
    polylines, size = render.layout(rows, line_height=64)
    sheet = render.to_image(polylines, size, stroke_width=1.9, dpi_scale=2.0)

    labelled = Image.new('RGBA', (sheet.width + 260, sheet.height), '#ffffff')
    labelled.alpha_composite(sheet, (260, 0))
    draw = ImageDraw.Draw(labelled)
    font = ImageFont.load_default(size=26)
    for i, style in enumerate(styles):
        y = (render.MARGIN + i * 64 - 10) * 2
        draw.text((24, y), '{:<2} {}'.format(style, style_name(style)), fill='#9a9a9a', font=font)
    labelled.convert('RGB').save(os.path.join(DOCS, 'styles.png'))

    for name in ('styles.png',):
        print(name, os.path.getsize(os.path.join(DOCS, name)) // 1024, 'KB')


if __name__ == '__main__':
    main()
