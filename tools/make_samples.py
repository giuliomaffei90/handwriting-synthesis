"""Dev tool: build the images used in readme.md."""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hw import render  # noqa: E402
from hw.engine import Model, available_styles  # noqa: E402

DOCS = os.path.join(ROOT, 'docs')


def main():
    os.makedirs(DOCS, exist_ok=True)
    model = Model()

    hero = model.generate(
        ['Handwriting, generated on a Mac', 'by a neural network from 2013,',
         'running on numpy alone.'],
        style=9, bias=0.85, seed=20240607)
    polylines, size = render.layout(hero)
    render.to_image(polylines, size, stroke_width=1.9, dpi_scale=2.0).save(
        os.path.join(DOCS, 'sample.png'))

    styles = available_styles()
    rows = [model.generate(['the quick brown fox jumps'], style=s, bias=0.85, seed=100 + s)[0]
            for s in styles]
    polylines, size = render.layout(rows, line_height=64)
    sheet = render.to_image(polylines, size, stroke_width=1.9, dpi_scale=2.0)

    labelled = Image.new('RGBA', (sheet.width + 70, sheet.height), '#ffffff')
    labelled.alpha_composite(sheet, (70, 0))
    draw = ImageDraw.Draw(labelled)
    font = ImageFont.load_default(size=26)
    for i, style in enumerate(styles):
        y = (render.MARGIN + i * 64 - 10) * 2
        draw.text((24, y), str(style), fill='#a3a3a3', font=font)
    labelled.convert('RGB').save(os.path.join(DOCS, 'styles.png'))

    for name in ('sample.png', 'styles.png'):
        print(name, os.path.getsize(os.path.join(DOCS, name)) // 1024, 'KB')


if __name__ == '__main__':
    main()
