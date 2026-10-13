import numpy as np

from src.data.pamap2 import _interp_nan, _runs, load_pamap2
from src.data.synthetic import write_fake_pamap2


def test_runs_and_interp():
    assert _runs(np.array([0, 0, 4, 4, 4, 0])) == [(0, 2), (2, 5), (5, 6)]
    x = np.array([[1.0], [np.nan], [3.0]])
    assert np.allclose(_interp_nan(x)[:, 0], [1, 2, 3])


def test_load_pamap2(tmp_path):
    write_fake_pamap2(tmp_path, n_subjects=2, seconds_per_activity=12)
    d = load_pamap2(tmp_path)
    assert d["X"].shape[1:] == (128, 6)
    assert np.isfinite(d["X"]).all()
    assert set(d["subjects"]) == {101, 102}
    assert len(d["classes"]) == 12
    # wrist acceleration converted to g: standing/walking magnitude ~1
    assert 0.7 < np.median(np.linalg.norm(d["X"][:, :, :3], axis=-1)) < 1.3
    hr = load_pamap2(tmp_path, include_hr=True)
    assert hr["channels"][-1] == "heart_rate" and hr["X"].shape[-1] == 7
