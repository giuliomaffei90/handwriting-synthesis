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


# The network emits pen positions about a unit apart, so drawing straight
# segments between them shows facets once the drawing is enlarged. Both
# renderers below round the corners the same way: every interior point becomes
# the control point of a quadratic curve running between its two neighbouring
# segment midpoints. to_svg emits those curves as real Beziers, to_image
# flattens them, so raster and vector output follow the identical path.

def _quad_segments(points):
    mids = (points[:-1] + points[1:]) / 2.0
    return mids[:-1], points[1:-1], mids[1:]


def smooth(points, samples=4):
    """Flatten the rounded path into a dense polyline (for rasterising)."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return points
    start, control, end = _quad_segments(points)
    t = np.linspace(0.0, 1.0, samples, endpoint=False).reshape(-1, 1, 1)
    curve = (1 - t) ** 2 * start + 2 * (1 - t) * t * control + t ** 2 * end
    return np.vstack([points[0], curve.transpose(1, 0, 2).reshape(-1, 2), points[-1]])


def svg_path(points):
    """The same rounded path as an SVG `d` string of quadratic Beziers."""
    points = np.asarray(points, dtype=np.float64)
    d = ['M{:.2f},{:.2f}'.format(*points[0])]
    if len(points) < 3:
        d += ['L{:.2f},{:.2f}'.format(*p) for p in points[1:]]
        return ' '.join(d)
    _, control, end = _quad_segments(points)
    d += ['Q{:.2f},{:.2f} {:.2f},{:.2f}'.format(c[0], c[1], e[0], e[1])
          for c, e in zip(control, end)]
    d.append('L{:.2f},{:.2f}'.format(*points[-1]))
    return ' '.join(d)


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


def to_svg(polylines, size, stroke_width=2.0, color='#111111', background=None):
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
        parts.append('<path d="{}"/>'.format(svg_path(stroke)))
    parts.append('</g></svg>')
    return '\n'.join(parts)


def resample(points, step):
    """Points along a polyline, no more than `step` apart (endpoints kept)."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points
    walked = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    if walked[-1] <= 0:
        return points[:1]
    at = np.append(np.arange(0.0, walked[-1], step), walked[-1])
    return np.stack([np.interp(at, walked, points[:, 0]),
                     np.interp(at, walked, points[:, 1])], axis=1)


def to_image(polylines, size, stroke_width=2.0, color='#111111', background=None,
             supersample=3, dpi_scale=1.0):
    """Render to a PIL image. `background=None` (the default) is transparent.

    The pen is stamped as overlapping dots along the path rather than drawn as
    a thick polyline: PIL's wide lines leave serrated edges at every vertex of
    a dense, hand-drawn path, while the union of the dots is exactly the stroke
    with round caps and joins.
    """
    from PIL import Image, ImageDraw

    factor = supersample * dpi_scale
    width = max(1, int(round(size[0] * factor)))
    height = max(1, int(round(size[1] * factor)))
    fill = background if background else (255, 255, 255, 0)
    image = Image.new('RGBA', (width, height), fill)
    draw = ImageDraw.Draw(image)

    radius = max(0.5, stroke_width * factor / 2.0)
    for stroke in polylines:
        points = resample(smooth(stroke) * factor, max(1.0, radius / 1.5))
        for x, y in points:
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

    out_size = (max(1, int(round(size[0] * dpi_scale))), max(1, int(round(size[1] * dpi_scale))))
    if supersample > 1:
        image = image.resize(out_size, Image.LANCZOS)
    return image
