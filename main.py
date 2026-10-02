"""Run all three pairs from any working directory. No high-level CV algorithms."""

import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse, json, time
from pathlib import Path
import numpy as np
from PIL import Image
from config import CONFIG
from camera import read_camera, ground_truth
from feature import detect, describe
from matching import match
from geometry import sampson, matrix_difference
from ransac import estimate
from visualize import corners, matches, epilines


def serial(value):
    """Convert numerical arrays for machine-readable result archive."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    """Extract features once; estimate both solvers on every image pair."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--materials',
        type=Path,
        default=Path(__file__).resolve().parent.parent / 'materials',
    )
    parser.add_argument(
        '--output', type=Path, default=Path(__file__).resolve().parent / 'output'
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    data = {}
    result = dict(config=CONFIG, images={}, pairs={})

    for index in (22, 23, 24):
        image = Image.open(args.materials / f'{index:08d}.jpg').convert('RGB')
        original = image.size
        scale = CONFIG['width'] / image.width
        image = image.resize(
            (CONFIG['width'], round(image.height * scale)), Image.Resampling.LANCZOS
        )
        gray = np.asarray(image.convert('L'), dtype=float) / 255
        pts = detect(gray, CONFIG)
        points, desc = describe(gray, pts, CONFIG)
        ext, k = read_camera(args.materials / f'{index:08d}_cam.txt')
        k = np.diag([scale, scale, 1]) @ k
        data[index] = (image, points, desc, ext, k)
        result['images'][str(index)] = dict(
            original_size=original,
            size=image.size,
            keypoints=len(pts),
            descriptors=len(desc),
            K=k,
            extrinsic=ext,
        )
        corners(image, pts, args.output / f'detected_features_{index}.jpg')
        print(f'{index}: {len(pts)} corners, {len(desc)} descriptors', flush=True)

    for i, j in ((22, 23), (23, 24), (22, 24)):
        a, p, da, ea, ka = data[i]
        b, q, db, eb, kb = data[j]
        ids, counts = match(da, db, CONFIG['ratio'])
        p = p[ids[:, 0]]
        q = q[ids[:, 1]]
        if len(p) < 8:
            raise RuntimeError(f'{i}-{j}: insufficient matches')
        tag = f'{i}_{j}'
        matches(a, b, p, q, args.output / f'matches_{tag}.jpg')
        models = {}
        for mode in ('7', '8'):
            model = estimate(p, q, ka, kb, mode, CONFIG)
            models[mode] = model
            print(
                f'{tag} {mode}-point: {len(p)} matches, '
                f'{model["inliers"]} inliers ({model["inlier_ratio"]:.1%}), '
                f'mean/median dS^2={model["mean_sampson"]:.3f}/'
                f'{model["median_sampson"]:.3f}, E inliers={model["E_inliers"]}',
                flush=True,
            )
            mask = model['inlier_mask']
            epilines(
                a,
                b,
                p[mask],
                q[mask],
                model['F'],
                args.output / f'epilines_{tag}_{mode}point.jpg',
            )

        # Selection uses calibrated residual consensus, never ground truth.
        selected = max(
            models,
            key=lambda mode: (models[mode]['E_inliers'], -models[mode]['E_median_all']),
        )
        model = models[selected]
        mask = sampson(model['F_from_E'], p, q) < CONFIG['threshold'] ** 2
        matches(a, b, p[mask], q[mask], args.output / f'inlier_matches_{tag}.jpg')
        epilines(
            a,
            b,
            p[mask],
            q[mask],
            model['F_from_E'],
            args.output / f'epilines_{tag}.jpg',
        )
        ce = sampson(model['F_from_E'], p, q)
        cm = ce < CONFIG['threshold'] ** 2
        epilines(
            a, b, p[cm], q[cm], model['F_from_E'], args.output / f'epilines_E_{tag}.jpg'
        )

        r, t, gt = ground_truth(ea, eb)
        gtf = np.linalg.inv(kb).T @ gt @ np.linalg.inv(ka)
        gt_errors = sampson(gtf, p, q)
        for m in models.values():
            m['E_gt_difference'] = matrix_difference(m['E'], gt)
        result['pairs'][tag] = dict(
            counts=counts,
            selected=selected,
            models=models,
            GT_R=r,
            GT_t=t,
            GT_E=gt,
            GT_median_all=float(np.median(gt_errors)),
            GT_median_selected_inliers=float(np.median(gt_errors[mask])),
        )
        np.savez(
            args.output / f'geometry_{tag}.npz',
            points1=p,
            points2=q,
            **{
                f'{key}_{mode}': value
                for mode, m in models.items()
                for key, value in m.items()
                if isinstance(value, np.ndarray)
            },
        )

    result['runtime_seconds'] = time.time() - start
    (args.output / 'results.json').write_text(
        json.dumps(result, default=serial, indent=2), encoding='utf-8'
    )
    lines = [
        'Two-view matching: true executed results',
        'Sampson errors are squared pixels in 1200x900 images.',
        json.dumps(CONFIG),
        'Extrinsic convention: world-to-camera. GT is evaluation only.',
    ]
    for tag, pair in result['pairs'].items():
        lines += [
            f'\nPAIR {tag}',
            f'''keypoints/descriptors: {[
                (
                    i,
                    result["images"][i]["keypoints"],
                    result["images"][i]["descriptors"],
                )
                for i in tag.split("_")
            ]}''',
            str(pair['counts']),
            f'selected: {pair["selected"]}-point',
        ]
        for mode, m in pair['models'].items():
            lines.append(f'{mode}-POINT')
            for k, v in m.items():
                if k.endswith('inlier_mask'):
                    continue
                lines.append(f'{k}:\n{v}' if isinstance(v, np.ndarray) else f'{k}: {v}')
        for key in (
            'GT_R',
            'GT_t',
            'GT_E',
            'GT_median_all',
            'GT_median_selected_inliers',
        ):
            lines.append(f'{key}:\n{pair[key]}')
    lines.append(f'Runtime seconds: {result["runtime_seconds"]:.2f}')
    (args.output / 'results.txt').write_text('\n'.join(lines), encoding='utf-8')
    print(
        f'Finished in {result["runtime_seconds"]:.1f}s. Outputs: {args.output}',
        flush=True,
    )


if __name__ == '__main__':
    main()
