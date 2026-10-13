"""Gravity-based alignment between device coordinate frames.

UCI HAR (smartphone on the waist) and SisFall (custom device on the waist)
point their axes differently. A model only works on data in the frame it
was trained in, so before feeding one sensor stream to both models we
estimate each frame's "up" direction from upright periods and rotate.
The same idea is a one-off calibration on a real device: stand still for
a few seconds, record gravity, done. Yaw (rotation about gravity) stays
unknown; rotation augmentation during training covers some of it.
"""
import numpy as np


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("zero vector")
    return v / n


def estimate_up(acc):
    """Mean gravity direction of (n, 3) or (n, T, 3) accelerometer data in g."""
    a = np.asarray(acc, dtype=float).reshape(-1, 3)
    return unit(a.mean(axis=0))


def rotation_between(a, b):
    """Rotation matrix R with R @ unit(a) == unit(b) (Rodrigues)."""
    a, b = unit(a), unit(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:                       # opposite: rotate 180 deg about any perpendicular axis
        axis = unit(np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0]))
        return 2 * np.outer(axis, axis) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def to_frame(acc, R):
    """Apply R to row-vector samples: (n, 3) @ R.T."""
    return np.asarray(acc) @ np.asarray(R).T
