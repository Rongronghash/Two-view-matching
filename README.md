# Two-View Matching

## 1. Project Overview
Course assignment implementing Harris detection, brute-force matching, normalized eight-point, seven-point, custom RANSAC and epipolar visualization. All five requirements in `../materials/task.txt` are implemented. No OpenCV is imported. This is Harris plus a local patch descriptor, **not SIFT**.

## 2. Dataset / Input
Three provided RGB images (22, 23, 24), each 1600×1200. Camera files contain labeled 4×4 `extrinsic` and 3×3 `intrinsic` matrices. `camera.py` parses both blocks without hardcoding K. Processing resizes to 1200×900 and scales K using diag(0.75, 0.75, 1). All reported F matrices, points and pixel errors refer to resized coordinates. To convert F to original coordinates, use F_original = S2.T @ F_resized @ S1.

We adopt world-to-camera convention Xc = R Xw + t. The file has no explicit direction label, so this is an assumption checked after fitting: GT epipolar residuals on final inliers are small (see results.txt). Relative pose is R21=R2 R1.T, t21=t2-R21 t1, E_GT=[t21]x R21. Rotations are used as supplied, with small text-rounding error. GT is called only after matching, RANSAC and model selection and cannot influence estimation.

## 3. Algorithm Pipeline
Load and resize → Harris → local patch → brute-force L2 → ratio test → mutual nearest → 7/8-point RANSAC → refit F → project E → constrained E optimization → consistent epilines. Process all three pairs.

## 4. Harris Corner Detector
Gaussian pre-smoothing sigma=1, central-difference image gradients, structure tensor with Gaussian sigma=1.5, R=det(M)-k trace(M)^2. Threshold positive responses relative to maximum, sort strongest first, then explicitly block neighboring pixels for greedy non-maximum suppression. Reject borders before describing points.

## 5. Patch Descriptor
Sample 25×25 support on a 13×13 grid after Gaussian smoothing. Subtract patch mean and divide by L2 norm (equivalent to variance normalization followed by constant scaling). Flatten to 169 dimensions. Reject near-flat patches. This handles affine brightness changes but offers no rotation, scale or perspective invariance.

## 6. Brute-force Matching + Ratio Test
Evaluate every squared L2 distance with matrix multiplication in blocks of 256. Obtain nearest and second nearest. Compare squared distances using ratio²=0.85², then require mutual nearest-neighbor consistency. Initial count is one nearest candidate per source descriptor. Ratio count is before cross check. Final count is the input to RANSAC. Plots sample at most 100 matches for readability, while fitting uses all matches.

## 7. Normalized 8-point Algorithm
Each point set gets Hartley normalization: subtract centroid and set mean radius to sqrt(2). Build A rows [u*x,u*y,u,v*x,v*y,v,x,y,1]. Last right singular vector gives F. Set the smallest singular value to zero, then denormalize F=T2.T F_normalized T1 and normalize its Frobenius norm. Detect rank-deficient samples. Minimal eight-point SVD uses the full right basis to preserve the null vector.

## 8. 7-point Algorithm
Exactly seven matches create a 7×9 design matrix and two right null vectors. Form F(alpha)=F1+alpha F2. Expand det(F(alpha)) with six signed permutations, multiplying three degree-one polynomials using convolution. Thus coefficients are directly constructed, rather than fitted from sample determinants. `numpy.roots` solves the cubic. Keep real roots, denormalize and enforce rank two. The implementation also handles the limiting root at infinity. Synthetic tests check recovery of a known nonplanar scene and sample epipolar constraints.

## 9. RANSAC
Uniform sampling without replacement and fixed seed, evaluate every seven-point candidate. Inlier test is squared Sampson distance < 1.5² px². Rank candidates by consensus count, then truncated residual sum. New best models get up to three eight-point consensus refits; retain refits only if score improves. Adaptive stopping uses confidence 0.999, minimum 100 and maximum 5000 iterations. Both modes run independently. Seven-point mode uses eight-point refits because the consensus has more than seven matches.

## 10. Essential Matrix
Start with E=K2.T F K1, project SVD to diag(s,s,0). A pure projection initially degraded the fit severely. Therefore optimize E=[t]x R using handwritten Rodrigues rotation, normalized translation and **generic** scipy.optimize.least_squares. Initialize both rotation branches from SVD, minimize signed pixel Sampson residual with soft_l1 loss (scale 0.7, maximum 250 evaluations), update consensus three times. This is our own residual/model implementation, not a library essential/pose estimator. Translation scale is unobservable, and no pose cheirality recovery is claimed.

Recompute F_from_E=inv(K2).T E inv(K1) and its own mask and errors. F RANSAC statistics and constrained E statistics are reported separately. Choose the final mode by E consensus, breaking ties with all-match median residual. GT never selects models. Near ties between modes can differ by numerical precision.

## 11. Epipolar Geometry Visualization
Compute l2=F x1 and l1=F.T x2 explicitly. Intersect each line with image boundaries. Twelve seeded inlier points use identical colors and numbers. Canonical `epilines_PAIR.jpg` and `inlier_matches_PAIR.jpg` use the final constrained E's F and E consensus. `epilines_PAIR_7point/8point.jpg` use the corresponding unconstrained RANSAC F. `epilines_E_PAIR.jpg` repeats the calibrated geometry explicitly.

## 12. Project Structure
`main.py` orchestration, `config.py` parameters, `camera.py` parsing/GT, `feature.py` detection/description, `matching.py` matching, `geometry.py` solvers/metrics/calibrated refinement, `ransac.py` consensus, `visualize.py` drawing, `check.py` synthetic and output validation, `write_readme.py` real table generator, `build_report.mjs` PPT source, `requirements.txt`, `README.md`, `output/`, `two_view_matching_report.pptx`. Private report previews and package audit are in `.report-build/` and `.codex-finalizer/`.

## 13. Installation
Python 3.10+ recommended. From the parent directory:
```powershell
cd Two-view-matching
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
Linux/macOS: activate with `source .venv/bin/activate` instead. Only NumPy, SciPy and Pillow are needed for the matching pipeline. Existing local dependencies were available and the pipeline was actually executed without installation.

## 14. How to Run
```powershell
python main.py
python check.py
python write_readme.py
```
Paths derive from `__file__`, so `python /absolute/path/to/main.py` also works. Optional input/output paths: `python main.py --materials ../materials --output output`.
Expected console output: keypoint and descriptor counts, then six solver summaries with match counts, F consensus, F mean/median error and E consensus; finally execution time and output location. The main command regenerates experiments, not the PPT. The supplied PPT already contains the executed results. Its editable source `build_report.mjs` uses the Codex bundled @oai/artifact-tool runtime. To rebuild, set CODEX_RUNTIME to its dependency directory and PRESENTATION_SKILL to the installed presentations skill directory, then run the marker and Node script as documented in the script header. It is separate from the Python dependencies and not required to run the assignment.

## 15. Output Files
`detected_features_22/23/24.jpg`; each pair has `matches_PAIR.jpg`, `inlier_matches_PAIR.jpg`, `epilines_PAIR.jpg`, `epilines_E_PAIR.jpg`, and both per-solver F epiline images. `results.txt` contains counts, F/E matrices, constrained F, F and E metrics, iteration/candidate counts, GT relative R/t/E and GT comparisons. `results.json` archives all numerical records and masks. `geometry_PAIR.npz` stores matched points and numerical matrices/masks for verification. `two_view_matching_report.pptx` is the 14-slide submission deck at project root.

## 16. Experimental Results
All numbers below come from the executed program. Errors are **squared Sampson distances in resized-image px²**, not Euclidean distances or original-image errors. F and E have independently recomputed consensus sets.

| Pair | Solver | Matches | F inliers | F median px² | E inliers | E ratio | E mean / median px² | E/GT difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 22_23 | 7 | 816 | 661 | 0.0787 | 663 | 81.2% | 0.2296 / 0.0789 | 0.00816 |
| 22_23 | 8 | 816 | 666 | 0.0803 | 663 | 81.2% | 0.2296 / 0.0789 | 0.00816 |
| 23_24 | 7 | 930 | 769 | 0.0699 | 767 | 82.5% | 0.1808 / 0.0682 | 0.00475 |
| 23_24 | 8 | 930 | 759 | 0.1405 | 767 | 82.5% | 0.1808 / 0.0682 | 0.00475 |
| 22_24 | 7 | 197 | 84 | 0.2133 | 93 | 47.2% | 0.3779 / 0.1393 | 0.00764 |
| 22_24 | 8 | 197 | 96 | 0.1663 | 92 | 46.7% | 0.3299 / 0.1561 | 0.00457 |

Detected points / valid descriptors: 22 = 3266, 23 = 3500, 24 = 3500. Both counts are equal in this run. Selected solver modes: 22_23 = 8-point, 23_24 = 7-point, 22_24 = 7-point. Sign-invariant E/GT difference is min(||E/||E|| ± E_GT/||E_GT||||_F). This metric depends on scene geometry and is not an angular pose error. Runtime on this machine: 3.5 seconds (hardware/library dependent).

Adjacent views retain many consistent matches. Wider 22–24 loses support and has substantially lower consensus. GT errors on final consensus support the world-to-camera convention, but GT is not used to certify each correspondence. Seven- and eight-point starts converge to almost identical calibrated solutions on adjacent pairs.

Final configuration: `{"width": 1200, "harris_k": 0.04, "response_threshold": 0.001, "nms_radius": 4, "max_keypoints": 3500, "patch_radius": 12, "patch_samples": 13, "ratio": 0.85, "threshold": 1.5, "max_iterations": 5000, "confidence": 0.999, "seed": 42}`. Gradient/descriptor smoothing and optimizer settings are documented above. Fixed RNG ensures repeatability on a fixed environment; BLAS/SVD precision can alter a tie across platforms.

## 17. Limitations
Patch descriptors lack rotation/scale invariance and repeated windows/tiles can create false matches. Near-planar architectural surfaces can bias a fundamental estimate, making essential projection alone unreliable. Constrained refinement is a local optimizer with no global optimum guarantee. Sampson distance is a first-order approximation. RANSAC confidence is approximate under degenerate/repeated samples. No distortion model is supplied, and none is applied. Ground-truth extrinsic direction is assumed and supported by residual checks, not explicitly guaranteed by a dataset specification. Geometric consensus alone does not prove semantic match correctness. Feature detection and display use resized images and cap keypoints. This project estimates E, rather than claiming a unique physical pose or a 3D reconstruction.
