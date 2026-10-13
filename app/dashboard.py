"""SafeStep caregiver dashboard: replay a simulated day through the
deployed INT8 models and the safety monitor.

Run:  streamlit run app/dashboard.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os  # noqa: E402
from datetime import timedelta  # noqa: E402

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.demo.pipeline import StreamProcessor, har_agreement, load_classifier  # noqa: E402
from src.demo.session import load_session  # noqa: E402

st.set_page_config(page_title="SafeStep", layout="wide")

SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]
SEVERITY_COLOR = {"critical": "#b42318", "high": "#dc6803", "medium": "#ca8504",
                  "low": "#1570ef", "info": "#667085"}


@st.cache_data(show_spinner="Running the replay through both models...")
def run_replay(session_path, har_dir, fall_dir, _mtimes):
    session = load_session(session_path)
    proc = StreamProcessor(load_classifier(har_dir), load_classifier(fall_dir),
                           session["meta"].get("fall_frame_rotation"))
    res = proc.run(session)
    rows = pd.DataFrame(res["rows"])
    alerts = pd.DataFrame([a.to_dict() | {"segment": a.details.get("segment")} for a in res["alerts"]])
    if not alerts.empty:
        alerts["time"] = pd.to_datetime(alerts["time"], format="ISO8601")
    segs = [{"title": s["title"], "start": s["start"]} for s in session["segments"]]
    return rows, alerts, res["summary"], segs, session["meta"], proc.har.classes, \
        proc.safety_config.fall_threshold


def benchmark(model_dir):
    p = Path(model_dir) / "benchmark.json"
    if not p.exists():
        return None
    b = json.loads(p.read_text())
    v = b["variants"][b.get("recommended_variant", "onnx_int8")]
    return {"size_kb": v["size_kb"], "latency": v["latency_ms_p50"], "variant": b.get("recommended_variant")}


# ---------------------------------------------------------------- sidebar
st.sidebar.title("SafeStep")
st.sidebar.caption("Wearable activity recognition and fall alerts for dementia care")
OUT = os.getenv("SAFESTEP_OUTPUTS", "outputs")
if OUT != "outputs":
    st.sidebar.warning(f"Showing results from `{OUT}/`. Smoke-mode results come from synthetic data and mean nothing.")
session_path = st.sidebar.text_input("Replay session", f"{OUT}/demo/session.npz")
har_dir = st.sidebar.text_input("HAR model", f"{OUT}/export/deploy_har-cnn1d")
fall_dir = st.sidebar.text_input("Fall model", f"{OUT}/export/fall_acc_only-fall")

missing = [p for p in (session_path, Path(har_dir) / "meta.json", Path(fall_dir) / "meta.json")
           if not Path(p).exists()]
if missing:
    st.title("SafeStep caregiver dashboard")
    st.warning("Missing: " + ", ".join(str(m) for m in missing))
    st.markdown("Run the pipeline first:\n\n```bash\npython run_all.py          # real data\n"
                "python run_all.py --smoke  # synthetic data, plumbing check\n```")
    st.stop()

mtimes = tuple(Path(p).stat().st_mtime for p in (session_path, Path(har_dir) / "meta.json",
                                                  Path(fall_dir) / "meta.json"))
rows, alerts, summary, segs, meta, classes, thr = run_replay(session_path, har_dir, fall_dir, mtimes)

# ---------------------------------------------------------------- header
st.title("Caregiver dashboard")
st.caption(f"Simulated day replayed through the deployed models. Activities: UCI HAR subject "
           f"{meta['har_subject']}. Fall: SisFall {meta['fall_file']}. Both are held-out people "
           "when the pipeline is run end to end.")

counts = alerts["severity"].value_counts().to_dict() if not alerts.empty else {}
c = st.columns(5)
c[0].metric("Critical alerts", counts.get("critical", 0))
c[1].metric("High", counts.get("high", 0))
c[2].metric("Medium / low", counts.get("medium", 0) + counts.get("low", 0))
c[3].metric("HAR agreement with replay truth", f"{har_agreement(rows.to_dict('records'), classes):.0%}")
hb, fb = benchmark(har_dir), benchmark(fall_dir)
if hb and fb:
    c[4].metric("On-device models", f"{hb['size_kb'] + fb['size_kb']:.0f} KB")
    c[4].caption(f"{hb['latency'] + fb['latency']:.2f} ms per window, both models, 1 CPU thread")

# ---------------------------------------------------------------- alerts
st.subheader("Alerts")
if alerts.empty:
    st.info("No alerts raised.")
else:
    view = alerts.assign(clock=alerts["time"].dt.strftime("%H:%M:%S"),
                         part=alerts["segment"].map(lambda i: segs[int(i)]["title"]))
    view["rank"] = view["severity"].map(SEVERITY_ORDER.index)
    st.dataframe(
        view.sort_values("time")[["clock", "severity", "type", "message", "part"]]
        .style.map(lambda s: f"color: {SEVERITY_COLOR.get(s, '#000')}; font-weight: 600", subset=["severity"]),
        hide_index=True, width="stretch")

# ---------------------------------------------------------------- replay
st.subheader("Replay")
seg_titles = [f"{s['start']:%H:%M}  {s['title']}" for s in segs]
seg_id = st.selectbox("Part of the day", range(len(segs)), format_func=lambda i: seg_titles[i])
seg = rows[rows["segment"] == seg_id].reset_index(drop=True)
t0 = seg["time"].min().to_pydatetime().replace(microsecond=0)
t1 = seg["time"].max().to_pydatetime().replace(microsecond=0) + timedelta(seconds=1)
now = st.slider("Replay up to", min_value=t0, max_value=t1, value=t1,
                step=timedelta(seconds=5), format="HH:mm:ss")
cur = seg[seg["time"] <= now]
last = cur.iloc[-1]

s = st.columns(4)
s[0].metric("Detected activity", last["activity"].replace("_", " "))
s[0].caption(f"{last['confidence']:.0%} confident")
s[1].metric("Ground truth", str(last.get("truth", "n/a")).replace("_", " "))
s[2].metric("Fall probability", f"{last['fall_prob']:.2f}")
s[2].caption(f"alarm threshold {thr:.2f}")
s[3].metric("Motion", f"{last['motion']:.3f} g")
s[3].caption("below 0.03 g counts as no movement")

YAX = alt.Axis(minExtent=150, labelLimit=140)   # same y-axis width on every chart so time lines up
base = alt.Chart(cur).encode(x=alt.X("time:T", title=None, scale=alt.Scale(domain=[t0, t1])))
order = list(classes) + ["fall"]
act = base.mark_tick(thickness=3, size=14).encode(
    y=alt.Y("activity:N", sort=order, title="Predicted", axis=YAX),
    color=alt.Color("activity:N", legend=None, scale=alt.Scale(scheme="tableau10")),
    tooltip=["time:T", "activity", alt.Tooltip("confidence:Q", format=".2f"), "truth"])
truth = base.mark_tick(thickness=3, size=14, color="#98a2b3").encode(
    y=alt.Y("truth:N", sort=order, title="Ground truth", axis=YAX))
st.altair_chart(act.properties(height=150), width="stretch")
st.altair_chart(truth.properties(height=120), width="stretch")

fall = base.mark_line(color="#b42318").encode(y=alt.Y("fall_prob:Q", title="Fall probability",
                                                       scale=alt.Scale(domain=[0, 1]), axis=YAX))
rule = alt.Chart(pd.DataFrame({"y": [thr]})).mark_rule(strokeDash=[4, 4], color="#667085").encode(y="y:Q")
motion = base.mark_area(opacity=0.4, color="#1570ef").encode(y=alt.Y("motion:Q", title="Motion (g)", axis=YAX))
layers = [fall, rule]
if not alerts.empty:
    seg_alerts = alerts[(alerts["segment"] == seg_id) & (alerts["time"] <= pd.Timestamp(now))]
    if not seg_alerts.empty:
        layers.append(alt.Chart(seg_alerts).mark_rule(strokeWidth=2).encode(
            x="time:T", color=alt.Color("severity:N", scale=alt.Scale(
                domain=SEVERITY_ORDER, range=[SEVERITY_COLOR[k] for k in SEVERITY_ORDER]),
                legend=alt.Legend(orient="top", title="Alerts")),
            tooltip=["type", "message"]))
st.altair_chart(alt.layer(*layers).properties(height=200), width="stretch")
st.altair_chart(motion.properties(height=110), width="stretch")

# ---------------------------------------------------------------- summary
st.subheader("Daily summary")
mins = pd.DataFrame(list(summary["minutes_per_activity"].items()), columns=["activity", "minutes"])
st.altair_chart(alt.Chart(mins).mark_bar().encode(
    x=alt.X("minutes:Q", title="Minutes in replay"), y=alt.Y("activity:N", sort="-x", title=None),
    tooltip=["activity", "minutes"]).properties(height=220), width="stretch")
st.caption("Thresholds for alerts are defaults, not clinically derived. "
           "Replay stitches two public datasets; see README limitations.")
