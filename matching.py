"""Brute-force matching without library matchers."""

import numpy as np


def match(a, b, ratio):
    """Evaluate all L2 distances in blocks, then ratio test and mutual nearest."""
    if min(len(a), len(b)) < 2:
        raise ValueError('Need at least two descriptors')
    best = []
    second = []
    reverse_dist = np.full(len(b), np.inf)
    reverse = np.zeros(len(b), int)
    for start in range(0, len(a), 256):
        d = np.maximum(
            0,
            np.sum(a[start : start + 256] ** 2, 1)[:, None]
            + np.sum(b * b, 1)[None, :]
            - 2 * a[start : start + 256] @ b.T,
        )
        ids = np.argsort(d, axis=1)[:, :2]
        best.extend(ids[:, 0])
        second.extend(d[np.arange(len(d)), ids[:, 1]])
        v = d.min(axis=0)
        update = v < reverse_dist
        reverse[update] = start + d.argmin(axis=0)[update]
        reverse_dist[update] = v[update]
    best = np.array(best)
    second = np.array(second)
    first = np.sum((a - b[best]) ** 2, axis=1)
    keep = first < ratio**2 * second
    mutual = reverse[best] == np.arange(len(a))
    rows = np.flatnonzero(keep & mutual)
    return np.column_stack([rows, best[rows]]), dict(
        initial=len(a), ratio_test=int(keep.sum()), cross_check=len(rows)
    )
