from datetime import datetime, timedelta

from src.safety import Observation, SafetyConfig, SafetyMonitor, summarize_day


def seq(start, steps, step_s, activity, fall_prob=0.0, motion=0.2):
    return [Observation(start + timedelta(seconds=i * step_s), activity, fall_prob, motion)
            for i in range(steps)]


def types(alerts):
    return [a.type for a in alerts]


DAY = datetime(2026, 1, 5, 14, 0)
NIGHT = datetime(2026, 1, 5, 2, 0)


def test_fall_without_recovery_escalates():
    obs = seq(DAY, 10, 1, "walking")
    t = obs[-1].time + timedelta(seconds=1)
    obs += [Observation(t, "lying", 0.97, 0.5)]
    obs += seq(t + timedelta(seconds=1), 45, 1, "lying", motion=0.005)
    out = types(SafetyMonitor().run(obs))
    assert out == ["FALL_SUSPECTED", "FALL_NO_RECOVERY"]


def test_fall_with_recovery():
    t = DAY
    obs = [Observation(t, "lying", 0.95, 0.6)]
    obs += seq(t + timedelta(seconds=1), 5, 1, "lying", motion=0.01)
    obs += seq(t + timedelta(seconds=6), 10, 1, "standing", motion=0.05)
    out = types(SafetyMonitor().run(obs))
    assert out == ["FALL_SUSPECTED", "FALL_RECOVERED"]


def test_daytime_inactivity_alerts_once():
    obs = seq(datetime(2026, 1, 5, 9, 0), 180, 60, "sitting", motion=0.005)  # 3 h
    out = types(SafetyMonitor().run(obs))
    assert out == ["INACTIVITY_CHECK"]


def test_night_stillness_is_sleep():
    obs = seq(datetime(2026, 1, 5, 23, 0), 400, 60, "lying", motion=0.002)
    assert SafetyMonitor().run(obs) == []


def test_night_wandering():
    obs = seq(NIGHT, 8 * 60, 1, "walking")
    assert "NIGHT_WANDERING" in types(SafetyMonitor().run(obs))


def test_daytime_walking_is_not_wandering():
    obs = seq(DAY, 8 * 60, 1, "walking")
    assert "NIGHT_WANDERING" not in types(SafetyMonitor().run(obs))


def test_restless_pacing_and_cooldown():
    obs, t = [], DAY
    for _ in range(14):                       # 14 bouts of 20 s walk + 60 s standing
        obs += seq(t, 20, 1, "walking")
        obs += seq(t + timedelta(seconds=20), 60, 1, "standing", motion=0.05)
        t += timedelta(seconds=80)
    out = types(SafetyMonitor().run(obs))
    assert out.count("RESTLESS_PACING") == 1  # 15 min cooldown, ~19 min of data


def test_threshold_is_configurable():
    obs = [Observation(DAY, "lying", 0.6, 0.5)]
    assert SafetyMonitor(SafetyConfig(fall_threshold=0.7)).run(obs) == []
    assert types(SafetyMonitor(SafetyConfig(fall_threshold=0.5)).run(obs)) == ["FALL_SUSPECTED"]


def test_summary():
    obs = seq(DAY, 61, 60, "walking")
    s = summarize_day(obs)
    assert s["minutes_per_activity"]["walking"] == 60.0


def test_gap_resets_open_fall():
    obs = [Observation(DAY, "lying", 0.95, 0.6)]
    obs += seq(DAY + timedelta(hours=3), 20, 1, "walking")   # device back 3 h later
    out = types(SafetyMonitor().run(obs))
    assert out == ["FALL_SUSPECTED"]                          # no stale "recovered"/"no recovery"


def test_summary_ignores_gaps():
    obs = seq(DAY, 11, 60, "walking") + seq(DAY + timedelta(hours=5), 11, 60, "sitting")
    s = summarize_day(obs)
    assert s["minutes_per_activity"] == {"sitting": 10.0, "walking": 10.0}
