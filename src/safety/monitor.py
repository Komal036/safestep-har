"""Caregiver safety rules on top of model outputs.

The models answer "what is the person doing in this window?" and
"does this window look like a fall?". Caregivers need different answers:
  * FALL_SUSPECTED      a fall-like event just happened
  * FALL_NO_RECOVERY    a fall followed by no movement (most urgent)
  * FALL_RECOVERED      the person got up again
  * INACTIVITY_CHECK    no movement for a long time during the day
  * NIGHT_WANDERING     sustained walking during night hours
  * RESTLESS_PACING     many short walking bouts in a short period (daytime)

SafetyMonitor is a streaming state machine: feed one Observation per
inference step, get back any alerts raised at that step. All thresholds
live in SafetyConfig so they can be tuned per patient.
"""
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

SEVERITY = {"FALL_NO_RECOVERY": "critical", "FALL_SUSPECTED": "high",
            "INACTIVITY_CHECK": "medium", "NIGHT_WANDERING": "medium",
            "RESTLESS_PACING": "low", "FALL_RECOVERED": "info"}


@dataclass
class Observation:
    time: datetime
    activity: str          # HAR model output, e.g. "walking", "lying"
    fall_prob: float       # fall model output in [0, 1]
    motion: float          # std of acceleration magnitude in the window (g)


@dataclass
class Alert:
    time: datetime
    type: str
    severity: str
    message: str
    details: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d["time"] = self.time.isoformat()
        return d


@dataclass
class SafetyConfig:
    fall_threshold: float = 0.5
    still_motion_g: float = 0.03            # below this the window counts as "no movement"
    fall_no_recovery_s: float = 30.0
    recovery_upright_s: float = 5.0
    inactivity_s: float = 2 * 3600          # daytime only
    night_start_hour: int = 22
    night_end_hour: int = 6
    wander_walk_s: float = 300.0            # walking time in the rolling window at night
    wander_window_s: float = 1800.0
    pacing_bouts: int = 6
    pacing_window_s: float = 1800.0
    min_bout_s: float = 10.0
    cooldown_s: float = 900.0               # per alert type, avoids alert storms
    max_gap_s: float = 300.0                # data gap longer than this resets short-term state
    walking_labels: tuple = ("walking", "walking_upstairs", "walking_downstairs")
    upright_labels: tuple = ("walking", "walking_upstairs", "walking_downstairs", "standing")


class SafetyMonitor:
    def __init__(self, config: SafetyConfig | None = None):
        self.cfg = config or SafetyConfig()
        self.reset()

    def reset(self):
        self.last_time = None
        self._reset_short_term()
        self.bouts = deque()          # finished walking bouts: (start, end)
        self.last_alert = {}

    def _reset_short_term(self):
        """Forget state that only makes sense for a continuous stream
        (device removed, out of range, or replay segments far apart)."""
        self.still_since = None
        self.inactivity_alerted = False
        self.fall_time = None
        self.fall_escalated = False
        self.upright_since = None
        self.bout_start = None

    # helpers
    def is_night(self, t: datetime):
        h, c = t.hour, self.cfg
        if c.night_start_hour > c.night_end_hour:
            return h >= c.night_start_hour or h < c.night_end_hour
        return c.night_start_hour <= h < c.night_end_hour

    def _emit(self, alerts, t, kind, message, cooldown=True, **details):
        if cooldown:
            last = self.last_alert.get(kind)
            if last is not None and (t - last).total_seconds() < self.cfg.cooldown_s:
                return
        self.last_alert[kind] = t
        alerts.append(Alert(t, kind, SEVERITY[kind], message, details))

    def _walking_seconds(self, now, window_s):
        lo = now - timedelta(seconds=window_s)
        spans = list(self.bouts)
        if self.bout_start is not None:
            spans.append((self.bout_start, now))
        return sum(max(0.0, (min(e, now) - max(s, lo)).total_seconds()) for s, e in spans)

    def update(self, obs: Observation):
        c, t, alerts = self.cfg, obs.time, []
        if self.last_time is not None and t < self.last_time:
            raise ValueError("observations must be in time order")
        if self.last_time is not None and (t - self.last_time).total_seconds() > c.max_gap_s:
            self._reset_short_term()
        self.last_time = t

        # stillness tracking
        if obs.motion < c.still_motion_g:
            if self.still_since is None:
                self.still_since = t
        else:
            self.still_since = None
            self.inactivity_alerted = False
        still_s = (t - self.still_since).total_seconds() if self.still_since else 0.0

        # upright tracking (for fall recovery)
        if obs.activity in c.upright_labels and obs.motion >= c.still_motion_g / 2:
            self.upright_since = self.upright_since or t
        else:
            self.upright_since = None

        # falls
        if obs.fall_prob >= c.fall_threshold and self.fall_time is None:
            self.fall_time, self.fall_escalated = t, False
            self._emit(alerts, t, "FALL_SUSPECTED", "Possible fall detected.",
                       cooldown=False, fall_prob=round(float(obs.fall_prob), 3))
        elif self.fall_time is not None:
            since_fall = (t - self.fall_time).total_seconds()
            if (self.upright_since is not None
                    and (t - self.upright_since).total_seconds() >= c.recovery_upright_s):
                self._emit(alerts, t, "FALL_RECOVERED", "Person is up and moving after the fall.",
                           cooldown=False, seconds_after_fall=round(since_fall, 1))
                self.fall_time = None
            elif (not self.fall_escalated and self.still_since is not None
                  and (t - max(self.still_since, self.fall_time)).total_seconds() >= c.fall_no_recovery_s):
                self.fall_escalated = True
                self._emit(alerts, t, "FALL_NO_RECOVERY",
                           "Fall followed by no movement. Check on the person now.",
                           cooldown=False, seconds_still=round(still_s, 1))

        # long inactivity (daytime only, night stillness is usually sleep)
        if (not self.is_night(t) and not self.inactivity_alerted
                and still_s >= c.inactivity_s and self.fall_time is None):
            self.inactivity_alerted = True
            self._emit(alerts, t, "INACTIVITY_CHECK",
                       f"No movement for {still_s / 3600:.1f} h.", cooldown=False,
                       minutes_still=round(still_s / 60, 1))

        # walking bouts
        walking = obs.activity in c.walking_labels
        if walking and self.bout_start is None:
            self.bout_start = t
        elif not walking and self.bout_start is not None:
            if (t - self.bout_start).total_seconds() >= c.min_bout_s:
                self.bouts.append((self.bout_start, t))
            self.bout_start = None
        horizon = t - timedelta(seconds=max(c.wander_window_s, c.pacing_window_s))
        while self.bouts and self.bouts[0][1] < horizon:
            self.bouts.popleft()

        if self.is_night(t):
            walk_s = self._walking_seconds(t, c.wander_window_s)
            if walk_s >= c.wander_walk_s:
                self._emit(alerts, t, "NIGHT_WANDERING",
                           f"{walk_s / 60:.0f} min of walking at night.",
                           walking_minutes=round(walk_s / 60, 1))
        else:
            lo = t - timedelta(seconds=c.pacing_window_s)
            n_bouts = sum(1 for s, e in self.bouts if e >= lo)
            if n_bouts >= c.pacing_bouts:
                self._emit(alerts, t, "RESTLESS_PACING",
                           f"{n_bouts} separate walking bouts in {c.pacing_window_s / 60:.0f} min.",
                           bouts=n_bouts)
        return alerts

    def run(self, observations):
        alerts = []
        for o in observations:
            alerts.extend(self.update(o))
        return alerts


def summarize_day(observations, alerts=(), max_gap_s=300.0):
    """Caregiver-facing summary: time per activity and alert counts.
    Gaps longer than max_gap_s (no data) are not credited to any activity."""
    obs = list(observations)
    time_per = {}
    for a, b in zip(obs, obs[1:]):
        dt = (b.time - a.time).total_seconds()
        if dt > max_gap_s:
            continue
        time_per[a.activity] = time_per.get(a.activity, 0.0) + dt
    counts = {}
    for al in alerts:
        counts[al.type] = counts.get(al.type, 0) + 1
    return {
        "minutes_per_activity": {k: round(v / 60, 1) for k, v in sorted(time_per.items())},
        "alerts": counts,
        "start": obs[0].time.isoformat() if obs else None,
        "end": obs[-1].time.isoformat() if obs else None,
    }
