"""Pillow drawing with handwritten line clipping and epipolar equations."""

import numpy as np
from PIL import Image, ImageDraw
from geometry import homogeneous


def corners(image, p, path):
    """Overlay detected corners."""
    out = image.copy()
    d = ImageDraw.Draw(out)
    for x, y in p:
        d.ellipse((x - 2, y - 2, x + 2, y + 2), outline='#00ff70', width=1)
    out.save(path)


def canvas(a, b):
    """Join images without changing their geometry."""
    out = Image.new('RGB', (a.width + b.width, max(a.height, b.height)))
    out.paste(a, (0, 0))
    out.paste(b, (a.width, 0))
    return out


def matches(a, b, p, q, path, seed=42):
    """Draw at most 100 reproducibly sampled correspondences."""
    out = canvas(a, b)
    d = ImageDraw.Draw(out)
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(p), min(100, len(p)), replace=False)
    for i in ids:
        color = tuple(rng.integers(60, 255, 3))
        x, y = p[i]
        u, v = q[i]
        u += a.width
        d.line((x, y, u, v), fill=color, width=1)
        for cx, cy in ((x, y), (u, v)):
            d.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), outline=color, width=2)
    out.save(path)


def clipped(line, w, h):
    """Intersect ax+by+c=0 with four image edges."""
    a, b, c = line
    pts = []
    if abs(b) > 1e-12:
        for x in (0, w - 1):
            y = -(a * x + c) / b
            if 0 <= y < h:
                pts.append((x, y))
    if abs(a) > 1e-12:
        for y in (0, h - 1):
            x = -(b * y + c) / a
            if 0 <= x < w:
                pts.append((x, y))
    return pts[:2]


def epilines(a, b, p, q, f, path, seed=42):
    """Draw l2=F x1 and l1=F^T x2, 12 matching numbered points."""
    out = canvas(a, b)
    d = ImageDraw.Draw(out)
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(p), min(12, len(p)), replace=False)
    for number, i in enumerate(ids, 1):
        color = tuple(rng.integers(50, 255, 3))
        for point, line, img, offset in (
            (p[i], f.T @ homogeneous(q[i : i + 1])[0], a, 0),
            (q[i], f @ homogeneous(p[i : i + 1])[0], b, a.width),
        ):
            ends = clipped(line, img.width, img.height)
            if len(ends) == 2:
                d.line([(x + offset, y) for x, y in ends], fill=color, width=2)
            x, y = point
            x += offset
            d.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=3)
            d.text(
                (x + 7, y - 12),
                str(number),
                fill=color,
                stroke_width=1,
                stroke_fill='black',
            )
    out.save(path)
