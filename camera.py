"""Camera parsing and evaluation only. Extrinsics never enter estimation."""

import numpy as np


def read_camera(path):
    """Parse labeled extrinsic (4x4) and intrinsic (3x3) blocks."""
    lines = [s.strip() for s in path.read_text().splitlines() if s.strip()]
    e, k = lines.index('extrinsic'), lines.index('intrinsic')
    return (
        np.array([[float(x) for x in s.split()] for s in lines[e + 1 : e + 5]]),
        np.array([[float(x) for x in s.split()] for s in lines[k + 1 : k + 4]]),
    )


def ground_truth(a, b):
    """Assume X_camera=R X_world+t; return relative R,t,E."""
    r = b[:3, :3] @ a[:3, :3].T
    t = b[:3, 3] - r @ a[:3, 3]
    x, y, z = t
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    e = skew @ r
    return r, t, e / np.linalg.norm(e)
