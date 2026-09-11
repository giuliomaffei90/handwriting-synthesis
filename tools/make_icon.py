"""Dev tool: build assets/AppIcon.icon, the app's Liquid Glass icon.

The icon is the model's own handwriting - "Aa", written by style 9 - laid out
as an Icon Composer document: a paper-coloured background and one layer of ink
that the system renders as Liquid Glass, with its specular highlights, its
translucency and its dark, tinted and clear appearances. Open the result in
Icon Composer to adjust it; tools/build_app.sh compiles it into the app.

Run: .venv/bin/python tools/make_icon.py
"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hw import render  # noqa: E402
from hw.engine import Model  # noqa: E402

CANVAS = 1024          # Icon Composer draws icons on a 1024 point square
GLYPH_WIDTH = 0.56     # of the canvas
PEN = 19.0             # points; the 1.0 icon used 12


def ink_image():
    """The pen strokes of "Aa", fitted and centred on the canvas, as a PNG.

    Icon Composer takes an SVG layer's paths as filled shapes and ignores
    their stroke, which turns handwriting into solid wedges - so the layer is
    drawn with the app's own renderer instead, ink on transparency, and the
    material is applied to exactly the shape the pen made.
    """
    strokes = Model().generate(['Aa'], style=9, bias=1.2, seed=11)
    polylines, page = render.layout(strokes, margin=0)
    scale = CANVAS * GLYPH_WIDTH / page[0]
    dx = (CANVAS - page[0] * scale) / 2
    dy = (CANVAS - page[1] * scale) / 2
    placed = [stroke * scale + [dx, dy] for stroke in polylines]
    return render.to_image(placed, (CANVAS, CANVAS), stroke_width=PEN, color='#000000',
                           background=None, supersample=3, dpi_scale=1.0)


def gradient(top, bottom):
    return {
        'linear-gradient': [top, bottom],
        'orientation': {'start': {'x': 0.5, 'y': 0}, 'stop': {'x': 0.5, 'y': 1}},
    }


def document():
    return {
        # paper by day, a sheet of slate by night
        'fill-specializations': [
            {'value': gradient('display-p3:0.99000,0.98400,0.96800,1.00000',
                               'display-p3:0.90500,0.89500,0.87000,1.00000')},
            {'appearance': 'dark',
             'value': gradient('display-p3:0.23000,0.23000,0.25000,1.00000',
                               'display-p3:0.09000,0.09000,0.10000,1.00000')},
        ],
        'groups': [
            {
                'layers': [
                    {
                        'fill-specializations': [
                            {'value': {'automatic-gradient':
                                       'display-p3:0.10000,0.10500,0.13000,1.00000'}},
                            {'appearance': 'dark',
                             'value': {'automatic-gradient':
                                       'display-p3:0.95000,0.94000,0.91000,1.00000'}},
                        ],
                        'image-name': 'handwriting.png',
                        'name': 'handwriting',
                    }
                ],
                'lighting': 'individual',
                'position': {'scale': 1, 'translation-in-points': [0, 0]},
                'shadow': {'kind': 'layer-color', 'opacity': 0.5},
                'specular': True,
                'translucency': {'enabled': True, 'value': 0.2},
            }
        ],
        'supported-platforms': {'squares': 'shared'},
    }


def main():
    icon = os.path.join(ROOT, 'assets', 'AppIcon.icon')
    shutil.rmtree(icon, ignore_errors=True)
    os.makedirs(os.path.join(icon, 'Assets'))
    ink_image().save(os.path.join(icon, 'Assets', 'handwriting.png'))
    with open(os.path.join(icon, 'icon.json'), 'w') as handle:
        json.dump(document(), handle, indent=2)
    print('wrote', icon)


if __name__ == '__main__':
    main()
