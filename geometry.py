"""Handwritten normalized 7/8-point solvers and epipolar metrics."""

import numpy as np
from itertools import permutations


def homogeneous(p):
    """Append homogeneous coordinate one."""
    return np.column_stack([p, np.ones(len(p))])


def normalize(p):
    """Hartley centering and mean radius sqrt(2)."""
    c = p.mean(axis=0)
    radius = np.linalg.norm(p - c, axis=1).mean()
    if radius < 1e-9:
        raise ValueError('Coincident points')
    s = np.sqrt(2) / radius
    t = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.]])
    return (homogeneous(p) @ t.T)[:, :2], t


def design(p, q):
    """Rows satisfy x2^T F x1=0 in row-major F convention."""
    x, y = p.T
    u, v = q.T
    return np.column_stack([u * x, u * y, u, v * x, v * y, v, x, y, np.ones(len(p))])


def rank2(f):
    """Project a matrix onto rank two and unit Frobenius norm."""
    u, s, v = np.linalg.svd(f)
    s[-1] = 0
    f = (u * s) @ v
    return f / np.linalg.norm(f)


def eight_point(p, q):
    """Hartley normalized linear solver, rank two before denormalization."""
    if len(p) < 8:
        raise ValueError('Eight correspondences required')
    p, t = normalize(p)
    q, w = normalize(q)
    a = design(p, q)
    if np.linalg.matrix_rank(a) < 8:
        raise ValueError('Degenerate design')
    _, _, v = np.linalg.svd(a, full_matrices=len(p) < 9)
    f = rank2(v[-1].reshape(3, 3))
    f = w.T @ f @ t
    return f / np.linalg.norm(f)


def seven_point(p, q):
    """Two null vectors; expand determinant cubic by signed permutations."""
    if len(p) != 7:
        raise ValueError('Exactly seven correspondences required')
    p, t = normalize(p)
    q, w = normalize(q)
    a = design(p, q)
    if np.linalg.matrix_rank(a) < 7:
        return []
    _, _, v = np.linalg.svd(a, full_matrices=True)
    f1, f2 = v[-1].reshape(3, 3), v[-2].reshape(3, 3)
    # Product of three linear polynomials, summed over determinant permutations.
    coefficients = np.zeros(4)
    for perm in permutations(range(3)):
        sign = (-1) ** sum(perm[i] > perm[j] for i in range(3) for j in range(i + 1, 3))
        poly = np.array([1.])
        for row, col in enumerate(perm):
            poly = np.convolve(poly, [f1[row, col], f2[row, col]])
        coefficients += sign * poly
    roots = np.roots(np.trim_zeros(coefficients[::-1], 'f'))
    result = []
    for root in roots:
        if abs(root.imag) < 1e-7 * (1 + abs(root.real)):
            result.append(rank2(w.T @ (f1 + root.real * f2) @ t))
    # A root at infinity occurs when det(F2)=0.
    if abs(coefficients[-1]) < 1e-12:
        result.append(rank2(w.T @ f2 @ t))
    return result


def sampson(f, p, q):
    """Squared first-order geometric error in pixel squared units."""
    x, y = homogeneous(p), homogeneous(q)
    fx = x @ f.T
    fy = y @ f
    numerator = np.sum(y * fx, axis=1) ** 2
    denominator = np.sum(fx[:, :2] ** 2, axis=1) + np.sum(fy[:, :2] ** 2, axis=1)
    return numerator / np.maximum(denominator, 1e-20)


def essential(f, k1, k2):
    """Enforce equal nonzero singular values on K2^T F K1."""
    u, s, v = np.linalg.svd(k2.T @ f @ k1)
    a = (s[0] + s[1]) / 2
    e = (u * np.array([a, a, 0])) @ v
    return e / np.linalg.norm(e)


def matrix_difference(a, b):
    """Scale-normalized Frobenius difference invariant to matrix sign."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(min(np.linalg.norm(a - b), np.linalg.norm(a + b)))


def skew(t):
    """Cross-product matrix."""
    x, y, z = t
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def rodrigues(v):
    """Handwritten exponential map from rotation vector to SO(3)."""
    theta = np.linalg.norm(v)
    k = skew(v)
    if theta < 1e-8:
        return np.eye(3) + k + 0.5 * k @ k
    return (
        np.eye(3) + np.sin(theta) / theta * k + (1 - np.cos(theta)) / theta**2 * k @ k
    )


def refine_essential(e, k1, k2, p, q, mask, threshold):
    """Optimize E=[t]x R using generic least squares, no pose library solver.

    Initialize both rotation branches of the E decomposition. Translation is
    normalized internally. Robust signed pixel Sampson residuals guide fitting.
    """
    from scipy.optimize import least_squares

    u, _, v = np.linalg.svd(e)
    if np.linalg.det(u) < 0:
        u[:, -1] *= -1
    if np.linalg.det(v) < 0:
        v[-1] *= -1
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
    ik1, ik2 = np.linalg.inv(k1), np.linalg.inv(k2)
    x, y = homogeneous(p), homogeneous(q)
    best = None
    for base in (u @ w @ v, u @ w.T @ v):

        def matrices(params):
            t = params[3:]
            t = t / max(np.linalg.norm(t), 1e-12)
            ee = skew(t) @ rodrigues(params[:3]) @ base
            ff = ik2.T @ ee @ ik1
            return ee / np.linalg.norm(ee), ff / np.linalg.norm(ff)

        def residual(params, active):
            _, f = matrices(params)
            fx = x[active] @ f.T
            fy = y[active] @ f
            den = np.sum(fx[:, :2] ** 2, 1) + np.sum(fy[:, :2] ** 2, 1)
            return np.sum(y[active] * fx, 1) / np.sqrt(np.maximum(den, 1e-20))

        params = np.r_[np.zeros(3), u[:, -1]]
        active = mask.copy()
        for _ in range(3):
            fit = least_squares(
                residual,
                params,
                args=(active,),
                loss='soft_l1',
                f_scale=0.7,
                max_nfev=250,
            )
            params = fit.x
            ee, ff = matrices(params)
            err = sampson(ff, p, q)
            active = err < threshold**2
            if active.sum() < 8:
                break
        score = (int(active.sum()), -float(np.minimum(err, threshold**2).sum()))
        if best is None or score > best[0]:
            best = (score, ee, ff)
    return best[1:]
