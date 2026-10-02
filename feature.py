"""Handwritten Harris detection and local intensity descriptors."""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def detect(gray, cfg):
    """Central differences, Gaussian structure tensor, threshold, greedy NMS."""
    g = gaussian_filter(gray.astype(float), 1.0)
    ix = np.zeros_like(g)
    iy = np.zeros_like(g)
    ix[:, 1:-1] = (g[:, 2:] - g[:, :-2]) / 2
    iy[1:-1, :] = (g[2:, :] - g[:-2, :]) / 2
    a, b, c = [gaussian_filter(v, 1.5) for v in (ix * ix, iy * iy, ix * iy)]
    response = a * b - c * c - cfg['harris_k'] * (a + b) ** 2
    border = cfg['patch_radius'] + 2
    response[:border, :] = 0
    response[-border:, :] = 0
    response[:, :border] = 0
    response[:, -border:] = 0
    yy, xx = np.where(response > cfg['response_threshold'] * response.max())
    order = np.argsort(-response[yy, xx], kind='stable')
    blocked = np.zeros(g.shape, bool)
    pts = []
    rad = cfg['nms_radius']
    for j in order:
        x, y = int(xx[j]), int(yy[j])
        if blocked[y, x]:
            continue
        pts.append((x, y))
        blocked[y - rad : y + rad + 1, x - rad : x + rad + 1] = True
        if len(pts) == cfg['max_keypoints']:
            break
    return np.array(pts, dtype=float).reshape(-1, 2)


def describe(gray, points, cfg):
    """Sample 25x25 support to 13x13, zero mean, unit L2; reject flat patches."""
    axis = np.linspace(-cfg['patch_radius'], cfg['patch_radius'], cfg['patch_samples'])
    dx, dy = np.meshgrid(axis, axis)
    smooth = gaussian_filter(gray.astype(float), 1.0)
    coords = [points[:, 1, None] + dy.ravel(), points[:, 0, None] + dx.ravel()]
    patches = map_coordinates(smooth, coords, order=1, mode='reflect')
    patches -= patches.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(patches, axis=1)
    valid = norm > 0.02
    return points[valid], (patches[valid] / norm[valid, None]).astype(np.float32)
