"""Turn generated stroke offsets into polylines, SVG and PNG.

The per-line transform (scale, denoise, slant correction, y flip) is the one
from the upstream demo; the layout is redone so the canvas is cropped to the
text instead of a fixed 1000px box.
"""
import numpy as np

from . import drawing

LINE_HEIGHT = 60.0
SCALE = 1.5
MARGIN = 24.0


def line_polylines(offsets, scale=SCALE, denoise=True, align=True):
    """[n, 3] offsets -> list of [k, 2] point arrays (one per pen stroke)."""
    if len(offsets) == 0:
        return []
    offsets = np.array(offsets, dtype=np.float64)
    offsets[:, :2] *= scale
    coords = drawing.offsets_to_coords(offsets)
    if denoise and len(coords) > 1:
        coords = drawing.denoise(coords)
    if align and len(coords) > 1:
        coords[:, :2] = drawing.align(coords[:, :2])
    coords[:, 1] *= -1

    breaks = np.where(coords[:-1, 2] == 1.0)[0] + 1
    return [s[:, :2] for s in np.split(coords, breaks) if len(s)]


def layout(lines_offsets, line_height=LINE_HEIGHT, scale=SCALE, align_mode='left',
           margin=MARGIN, denoise=True, align=True):
    """Stack per-line strokes into one page.

    Returns (polylines, (width, height)) in SVG user units.
    """
    per_line = [line_polylines(o, scale=scale, denoise=denoise, align=align)
                for o in lines_offsets]

    widths = []
    for strokes in per_line:
        if strokes:
            xs = np.concatenate([s[:, 0] for s in strokes])
            widths.append(xs.max() - xs.min())
        else:
            widths.append(0.0)
    page_width = max(widths) if widths else 0.0

    placed = []
    for i, strokes in enumerate(per_line):
        if not strokes:
            continue
        xs = np.concatenate([s[:, 0] for s in strokes])
        dx = -xs.min()
        if align_mode == 'center':
            dx += (page_width - widths[i]) / 2.0
        elif align_mode == 'right':
            dx += page_width - widths[i]
        dy = i * line_height
        placed.extend([s + np.array([dx, dy]) for s in strokes])

    if not placed:
        return [], (1.0, 1.0)

    all_points = np.concatenate(placed)
    lo = all_points.min(axis=0)
    placed = [s - lo + margin for s in placed]
    size = tuple(all_points.max(axis=0) - lo + 2 * margin)
    return placed, size


def to_svg(polylines, size, stroke_width=2.0, color='#111111', background='#ffffff'):
    width, height = size
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="{:.0f}" height="{:.0f}" '
        'viewBox="0 0 {:.2f} {:.2f}">'.format(width, height, width, height),
    ]
    if background:
        parts.append('<rect width="100%" height="100%" fill="{}"/>'.format(background))
    parts.append('<g fill="none" stroke="{}" stroke-width="{:.2f}" stroke-linecap="round" '
                 'stroke-linejoin="round">'.format(color, stroke_width))
    for stroke in polylines:
        d = 'M' + ' L'.join('{:.2f},{:.2f}'.format(x, y) for x, y in stroke)
        parts.append('<path d="{}"/>'.format(d))
    parts.append('</g></svg>')
    return '\n'.join(parts)


def to_image(polylines, size, stroke_width=2.0, color='#111111', background='#ffffff',
             supersample=3, dpi_scale=1.0):
    """Render to a PIL image. `background=None` gives a transparent PNG."""
    from PIL import Image, ImageDraw

    factor = supersample * dpi_scale
    width = max(1, int(round(size[0] * factor)))
    height = max(1, int(round(size[1] * factor)))
    fill = background if background else (255, 255, 255, 0)
    image = Image.new('RGBA', (width, height), fill)
    draw = ImageDraw.Draw(image)
    pen = max(1, int(round(stroke_width * factor)))
    for stroke in polylines:
        points = [(float(x) * factor, float(y) * factor) for x, y in stroke]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse([x - pen / 2, y - pen / 2, x + pen / 2, y + pen / 2], fill=color)
        else:
            draw.line(points, fill=color, width=pen, joint='curve')

    out_size = (max(1, int(round(size[0] * dpi_scale))), max(1, int(round(size[1] * dpi_scale))))
    if supersample > 1:
        image = image.resize(out_size, Image.LANCZOS)
    return image
