"""Synthetic mathematical checks and real output integrity audit."""

from pathlib import Path
import json
import numpy as np
from PIL import Image
from geometry import (
    eight_point,
    seven_point,
    rodrigues,
    skew,
    sampson,
    essential,
    matrix_difference,
)


def main():
    """Check solvers against noiseless nonplanar data and audited real files."""
    rng = np.random.default_rng(123)
    x = rng.normal(size=(40, 3))
    x[:, 2] += 6
    r = rodrigues(np.array([.06, -.09, .03]))
    t = np.array([.7, .1, .2])
    y = x @ r.T + t
    p = x[:, :2] / x[:, 2, None]
    q = y[:, :2] / y[:, 2, None]
    truth = skew(t) @ r
    f = eight_point(p, q)
    assert sampson(f, p, q).max() < 1e-20
    assert matrix_difference(f, truth) < 1e-8
    fs = seven_point(p[:7], q[:7])
    assert 1 <= len(fs) <= 3
    assert min(matrix_difference(v, truth) for v in fs) < 1e-7
    assert all(sampson(v, p[:7], q[:7]).max() < 1e-18 for v in fs)
    s = np.linalg.svd(essential(f, np.eye(3), np.eye(3)), compute_uv=False)
    assert abs(s[0] - s[1]) < 1e-10 and s[2] < 1e-10
    out = Path(__file__).resolve().parent / 'output'
    result = json.loads((out / 'results.json').read_text())
    for tag, pair in result['pairs'].items():
        data = np.load(out / f'geometry_{tag}.npz')
        for mode, m in pair['models'].items():
            f = np.array(m['F'])
            e = np.array(m['E'])
            assert np.isfinite(f).all() and np.isfinite(e).all()
            assert np.linalg.svd(f, compute_uv=False)[2] < 1e-10
            s = np.linalg.svd(e, compute_uv=False)
            assert abs(s[0] - s[1]) < 1e-10 and s[2] < 1e-10
            error = sampson(f, data['points1'], data['points2'])
            assert (
                int((error < result['config']['threshold'] ** 2).sum()) == m['inliers']
            )
            assert m['E_inliers'] >= 8
        for prefix in ('matches', 'inlier_matches', 'epilines', 'epilines_E'):
            with Image.open(out / f'{prefix}_{tag}.jpg') as im:
                im.verify()
    print(
        'PASS: synthetic 7/8-point, rank/essential constraints, '
        'real masks and output images'
    )


if __name__ == '__main__':
    main()
