import numpy as np
import pytest

PUZ = [int(c) for c in
       "530070000600195000098000060800060003400803001700020006060000280000419005000080079"]
SOL = [int(c) for c in
       "534678912672195348198342567859761423426853791713924856961537284287419635345286179"]


class FakeModel:
    """Test double: the first 9 pixels of every 'image' hold the class probabilities."""
    def predict_proba(self, images, calibrated=True):
        p = images[:, :9].astype(np.float64)
        return p / p.sum(1, keepdims=True)


def build_images(puzzle=PUZ, conf=0.97, misreads=None):
    """misreads: {cell: (wrong_digit, p_wrong, p_true)}"""
    misreads = misreads or {}
    imgs = np.zeros((81, 784), np.float32)
    for i, v in enumerate(puzzle):
        if not v:
            continue
        p = np.full(9, (1 - conf) / 8)
        p[v - 1] = conf
        if i in misreads:
            w, pw, pt = misreads[i]
            p[:] = (1 - pw - pt) / 7
            p[w - 1], p[v - 1] = pw, pt
        imgs[i, :] = 0.5
        imgs[i, :9] = p
    return imgs


@pytest.fixture
def model():
    return FakeModel()
