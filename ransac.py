"""Custom adaptive RANSAC with local refits, scoring in pixels."""

import numpy as np
from geometry import eight_point, seven_point, sampson, essential, refine_essential


def estimate(p, q, k1, k2, mode, cfg):
    """Evaluate every minimal candidate; return raw F and constrained E/F."""
    n = int(mode)
    rng = np.random.default_rng(cfg['seed'] + n)
    threshold = cfg['threshold'] ** 2
    best = None
    score = (-1, -np.inf)
    limit = cfg['max_iterations']
    iteration = 0
    candidates = 0

    def evaluate(f):
        error = sampson(f, p, q)
        mask = error < threshold
        return (
            (int(mask.sum()), -float(np.minimum(error, threshold).sum())),
            mask,
            error,
        )

    while iteration < limit:
        ids = rng.choice(len(p), n, replace=False)
        iteration += 1
        try:
            fs = (
                seven_point(p[ids], q[ids]) if n == 7 else [eight_point(p[ids], q[ids])]
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        for f in fs:
            candidates += 1
            sc, mask, err = evaluate(f)
            if sc > score:
                # Local optimization from consensus, independent of cameras/GT.
                for _ in range(3):
                    if mask.sum() < 8:
                        break
                    try:
                        refined = eight_point(p[mask], q[mask])
                    except (ValueError, np.linalg.LinAlgError):
                        break
                    new_sc, new_mask, new_err = evaluate(refined)
                    if new_sc <= sc:
                        break
                    f, sc, mask, err = refined, new_sc, new_mask, new_err
                best = (f, mask, err)
                score = sc
                prob = (mask.mean()) ** n
                if 0 < prob < 1:
                    limit = min(
                        limit,
                        max(
                            100,
                            int(
                                np.ceil(np.log(1 - cfg['confidence']) / np.log1p(-prob))
                            ),
                        ),
                    )
    if best is None or best[1].sum() < 8:
        raise RuntimeError('RANSAC has insufficient consensus')
    f, mask, err = best
    e = essential(f, k1, k2)
    e, calibrated = refine_essential(e, k1, k2, p, q, mask, cfg['threshold'])
    ce = sampson(calibrated, p, q)
    cm = ce < threshold
    # Keep explicit unconstrained vs constrained diagnostics, never mix their masks.
    return dict(
        F=f,
        E=e,
        F_from_E=calibrated,
        inlier_mask=mask,
        inliers=int(mask.sum()),
        inlier_ratio=float(mask.mean()),
        mean_sampson=float(err[mask].mean()),
        median_sampson=float(np.median(err[mask])),
        E_inlier_mask=cm,
        E_inliers=int(cm.sum()),
        E_mean_sampson=float(ce[cm].mean()) if cm.any() else None,
        E_inlier_ratio=float(cm.mean()),
        E_median_sampson=float(np.median(ce[cm])) if cm.any() else None,
        E_median_all=float(np.median(ce)),
        iterations=iteration,
        candidates=candidates,
    )
