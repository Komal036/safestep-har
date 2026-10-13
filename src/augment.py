"""Train-time augmentations for IMU windows (numpy, applied per batch).

Rotation matters most for wearables: the same activity produces different
axis readings depending on how the device sits on the wrist.
"""
import numpy as np


def xyz_groups(channels):
    """Indices of channel triplets sharing a prefix, e.g. body_acc_{x,y,z}."""
    groups = {}
    for i, c in enumerate(channels):
        if c[-2:] in ("_x", "_y", "_z"):
            groups.setdefault(c[:-2], {})[c[-1]] = i
    return [[g["x"], g["y"], g["z"]] for g in groups.values() if len(g) == 3]


def random_rotation(rng, max_deg):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    theta = np.deg2rad(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


class Augmenter:
    def __init__(self, channels, jitter_std=0.03, scale_std=0.1, rotate_deg=15.0, p=0.8, seed=0):
        self.groups = xyz_groups(channels)
        self.jitter_std, self.scale_std = jitter_std, scale_std
        self.rotate_deg, self.p = rotate_deg, p
        self.rng = np.random.default_rng(seed)

    def __call__(self, X):
        """X: (B, T, C) normalized batch. Returns an augmented copy."""
        X = X.copy()
        B = len(X)
        mask = self.rng.random(B) < self.p
        if self.rotate_deg > 0 and self.groups:
            for b in np.where(mask)[0]:
                R = random_rotation(self.rng, self.rotate_deg)  # same rotation for all sensors
                for g in self.groups:
                    X[b][:, g] = X[b][:, g] @ R.T
        if self.scale_std > 0:
            s = self.rng.normal(1.0, self.scale_std, size=(B, 1, X.shape[2]))
            X = np.where(mask[:, None, None], X * s, X)
        if self.jitter_std > 0:
            X = np.where(mask[:, None, None],
                         X + self.rng.normal(0, self.jitter_std, X.shape), X)
        return X.astype(np.float32)
