"""Stream-level fall detection and event metrics.

Window accuracy is a weak proxy for a fall alarm. What a caregiver feels is:
  * did the device alarm for each real fall?          -> event recall
  * how often does it alarm when nothing happened?    -> false alarms per hour
So we slide the detector over whole recordings, merge nearby positive
windows into alarm events, and score at the event level.
"""
import numpy as np
import torch

from src.data.sisfall import impact_index


def window_starts(n, window, step):
    if n < window:
        return np.array([0])
    return np.arange(0, n - window + 1, step)


def model_score_fn(model, norm, window, step, device="cpu", batch_size=512):
    """Return f(signal) -> (starts, fall_prob) using a trained window classifier."""
    def score(sig):
        starts = window_starts(len(sig), window, step)
        if len(sig) < window:
            sig = np.pad(sig, ((0, window - len(sig)), (0, 0)), mode="edge")
        wins = np.stack([sig[s:s + window] for s in starts])
        wins = norm(wins)
        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(wins), batch_size):
                xb = torch.from_numpy(wins[i:i + batch_size]).to(device)
                probs.append(torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy())
        return starts, np.concatenate(probs)
    return score


def threshold_score_fn(window, step, fs=50, impact_g=2.5, still_std_g=0.15, settle_s=0.5):
    """Classic rule-based detector: hard impact followed by stillness.

    Fires on a window when the acceleration magnitude peak exceeds impact_g
    and the signal after the impact (skipping settle_s) is nearly still.
    """
    settle = int(settle_s * fs)

    def score(sig):
        starts = window_starts(len(sig), window, step)
        mag = np.linalg.norm(sig[:, :3], axis=1)
        out = np.zeros(len(starts), dtype=np.float32)
        for i, s in enumerate(starts):
            m = mag[s:s + window]
            p = int(np.argmax(m))
            post = m[p + settle:]
            if m[p] >= impact_g and len(post) >= fs // 2 and post.std() < still_std_g:
                out[i] = 1.0
        return starts, out
    return score


def alarm_events(starts, probs, thr, window, refractory):
    """Merge positive windows into events. Returns list of (first_start, last_end)."""
    events = []
    for s, p in zip(starts, probs):
        if p < thr:
            continue
        if events and s - events[-1][2] <= refractory:
            events[-1][1] = s + window
            events[-1][2] = s
        else:
            events.append([s, s + window, s])  # [begin, end, last positive start]
    return [(b, e) for b, e, _ in events]


def score_trials(trials, score_fn):
    """Run the detector once per trial; thresholds can then be swept cheaply."""
    cache = []
    for t in trials:
        starts, probs = score_fn(t["signal"])
        cache.append({
            "starts": starts, "probs": probs, "is_fall": t["is_fall"],
            "elderly": t["elderly"], "n": len(t["signal"]), "fs": t["fs"],
            "peak": impact_index(t["signal"]) if t["is_fall"] else None,
        })
    return cache


def event_metrics(cache, thr, window, refractory_s=10.0, tolerance_s=1.0):
    n_falls = detected = 0
    fa = {"young": 0, "elderly": 0}
    hours = {"young": 0.0, "elderly": 0.0}
    for c in cache:
        fs = c["fs"]
        events = alarm_events(c["starts"], c["probs"], thr, window, int(refractory_s * fs))
        if c["is_fall"]:
            n_falls += 1
            tol = int(tolerance_s * fs)
            if any(b - tol <= c["peak"] < e + tol for b, e in events):
                detected += 1
        else:
            g = "elderly" if c["elderly"] else "young"
            fa[g] += len(events)
            hours[g] += c["n"] / fs / 3600
    total_h = hours["young"] + hours["elderly"]
    return {
        "threshold": float(thr),
        "n_falls": n_falls,
        "detected": detected,
        "recall": detected / n_falls if n_falls else float("nan"),
        "false_alarms": fa["young"] + fa["elderly"],
        "false_alarms_elderly": fa["elderly"],
        "adl_hours": total_h,
        "adl_hours_elderly": hours["elderly"],
        "fa_per_hour": (fa["young"] + fa["elderly"]) / total_h if total_h else float("nan"),
        "fa_per_hour_elderly": fa["elderly"] / hours["elderly"] if hours["elderly"] else float("nan"),
    }


def pick_threshold(cache, window, target_recall, grid=None, **kw):
    """Highest threshold that still reaches target recall (fewest false alarms).

    If no threshold reaches the target, take the best achievable recall,
    breaking ties by fewer false alarms.
    """
    grid = np.round(np.linspace(0.05, 0.99, 95), 3) if grid is None else grid
    rows = [event_metrics(cache, t, window, **kw) for t in grid]
    ok = [r for r in rows if r["recall"] >= target_recall]
    if ok:
        return max(ok, key=lambda r: r["threshold"])
    return max(rows, key=lambda r: (r["recall"], -r["false_alarms"]))
