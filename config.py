"""Reproducible experiment parameters, in resized image pixels."""

CONFIG = dict(
    width=1200,
    harris_k=0.04,
    response_threshold=0.001,
    nms_radius=4,
    max_keypoints=3500,
    patch_radius=12,
    patch_samples=13,
    ratio=0.85,
    threshold=1.5,
    max_iterations=5000,
    confidence=0.999,
    seed=42,
)
