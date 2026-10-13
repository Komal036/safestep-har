import numpy as np

from src.data.sisfall import (load_sisfall_trials, parse_file, subject_id, trials_to_windows)
from src.fall.stream import alarm_events, event_metrics, pick_threshold, score_trials, threshold_score_fn
from tests.sisfall_synth import make_sisfall


def test_parse_units(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("256, 0, 0, 16384, 0, 0, 1, 2, 3;\n 0,256,0,0,0,0,1,2,3;\n")
    sig = parse_file(p)
    assert sig.shape == (2, 6)
    assert np.isclose(sig[0, 0], 1.0)            # 256 counts = 1 g
    assert np.isclose(sig[0, 3], 1000.0)         # 16384 counts = 1000 deg/s


def test_subject_ids_do_not_collide():
    assert subject_id("SA", "01") == 1 and subject_id("SE", "01") == 101


def test_loader_and_windows(tmp_path):
    trials = load_sisfall_trials(make_sisfall(tmp_path, young=2, elderly=1, reps=1))
    assert {t["fs"] for t in trials} == {50}
    falls = [t for t in trials if t["is_fall"]]
    assert len(falls) == 2 and all(not t["elderly"] for t in falls)
    w = trials_to_windows(trials, window=128, adl_step=64, peak_positions=(40, 60))
    assert w["X"].shape[1:] == (128, 6)
    assert (w["y"] == 1).sum() == 4              # 2 falls x 2 shifts
    # impact sits at the requested position
    fall_win = w["X"][w["y"] == 1][0]
    assert abs(int(np.argmax(np.linalg.norm(fall_win[:, :3], axis=1))) - 40) <= 1


def test_alarm_merging():
    starts = np.arange(0, 1000, 25)
    probs = np.zeros(len(starts))
    probs[[4, 5, 6]] = 0.9        # one burst
    probs[[30]] = 0.9             # far away, separate event
    ev = alarm_events(starts, probs, 0.5, window=128, refractory=500)
    assert len(ev) == 2


def test_threshold_baseline_detects_synthetic_falls(tmp_path):
    trials = load_sisfall_trials(make_sisfall(tmp_path, young=3, elderly=1, reps=1))
    cache = score_trials(trials, threshold_score_fn(128, 25))
    m = event_metrics(cache, 0.5, 128)
    assert m["recall"] == 1.0
    assert m["false_alarms"] == 0
    assert pick_threshold(cache, 128, 0.9)["recall"] == 1.0
