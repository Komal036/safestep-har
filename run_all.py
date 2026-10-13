"""Run the whole SafeStep pipeline with one command (Windows, macOS, Linux).

    python run_all.py                    # real data, every stage
    python run_all.py --stages fall export demo
    python run_all.py --smoke            # synthetic data, ~2-4 min on CPU: checks all the plumbing
    python run_all.py --serve dashboard  # start the Streamlit dashboard
    python run_all.py --serve api        # start the FastAPI server (docs at /docs)

Real mode writes to outputs/. Smoke mode writes to outputs_smoke/ and uses
data/synthetic/, so it can never overwrite or be confused with real results.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
STAGES = ["data", "baseline", "deep", "ablations", "pamap2", "fall", "deploy", "export", "demo", "report"]

# config file -> stage
CONFIGS = {
    "baseline": ["configs/default.yaml"],
    "deep": ["configs/deep.yaml"],
    "ablations": ["configs/ablation_acc_only.yaml", "configs/ablation_no_augment.yaml"],
    "pamap2": ["configs/baseline_pamap2.yaml", "configs/deep_pamap2.yaml"],
    "fall": ["configs/fall.yaml", "configs/fall_acc_only.yaml"],
    "deploy": ["configs/deploy_har.yaml"],
}


def script_for(cfg_path):
    if "fall" in Path(cfg_path).name:
        return "src.train_fall"
    if "baseline" in Path(cfg_path).name or Path(cfg_path).name == "default.yaml":
        return "src.train_baseline"
    return "src.train_deep"


def sh(args, env=None):
    print("\n$ " + " ".join(str(a) for a in args), flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=ROOT,
                   env={**os.environ, **(env or {})})
    print(f"  done in {time.time() - t0:.0f}s", flush=True)


def smoke_config(cfg_path, out_root):
    """Same config, synthetic data, tiny training budget, separate output folder."""
    cfg = yaml.safe_load((ROOT / cfg_path).read_text())
    cfg["data_dir"] = "data/synthetic"
    cfg["output_dir"] = cfg["output_dir"].replace("outputs/", f"{out_root}/", 1)
    cfg["use_gpu"] = False
    if "cv" in cfg:
        cfg["cv"]["n_splits"] = 3
    if "train" in cfg:
        cfg["train"].update(epochs=10, patience=10, lr=0.003)
    for m in (cfg.get("models") or {}).values():
        if isinstance(m, dict) and "n_estimators" in m:
            m["n_estimators"] = 40
    path = ROOT / out_root / "configs" / Path(cfg_path).name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path.relative_to(ROOT)


def stage_data(smoke):
    if smoke:
        sys.path.insert(0, str(ROOT))
        from src.data.synthetic import write_all
        if not (ROOT / "data/synthetic/UCI HAR Dataset").exists():
            print("Writing synthetic datasets to data/synthetic/ ...")
            write_all(ROOT / "data/synthetic")
        return
    sh(["data/download.py", "--dataset", "uci_har"])
    sh(["data/download.py", "--dataset", "pamap2"])
    sh(["data/download.py", "--dataset", "sisfall"])
    if not any((ROOT / "data/raw/SisFall_dataset").glob("S*/*.txt")):
        sys.exit("\nSisFall is missing (manual download, instructions above). "
                 "Add it, then rerun: python run_all.py --stages fall deploy export demo report")


def train(stage, smoke, out_root):
    for cfg in CONFIGS[stage]:
        path = smoke_config(cfg, out_root) if smoke else cfg
        sh(["-m", script_for(cfg), "--config", path])


def stage_export(out_root, data_dir):
    cks = [f"{out_root}/deploy_har/cnn1d_last_fold.pt", f"{out_root}/fall_acc_only/fall_last_fold.pt",
           f"{out_root}/deep/cnn1d_last_fold.pt", f"{out_root}/fall/fall_last_fold.pt"]
    cks = [c for c in cks if (ROOT / c).exists()]
    if not cks:
        sys.exit("No checkpoints to export. Run the deploy and fall stages first.")
    sh(["-m", "src.export.onnx_export", "--checkpoint", *cks, "--data-dir", data_dir,
        "--out", f"{out_root}/export"])


def stage_demo(out_root, data_dir):
    sh(["-m", "src.demo.session", "--data-dir", data_dir, "--out", f"{out_root}/demo/session.npz",
        "--har-meta", f"{out_root}/export/deploy_har-cnn1d/meta.json",
        "--fall-meta", f"{out_root}/export/fall_acc_only-fall/meta.json"])


def stage_report(out_root, smoke):
    """Collect every results table into one file for the README."""
    parts = ["# SafeStep results\n"]
    if smoke:
        parts.append("> SMOKE RUN ON SYNTHETIC DATA. These numbers mean nothing; do not report them.\n")
    titles = {"baseline": "Feature baselines (UCI HAR)", "deep": "Deep models (UCI HAR)",
              "ablation_acc_only": "Ablation: accelerometer only", "ablation_no_augment": "Ablation: no augmentation",
              "baseline_pamap2": "Feature baselines (PAMAP2 wrist)", "deep_pamap2": "Deep models (PAMAP2 wrist)",
              "fall": "Fall detection (SisFall, acc + gyro)", "fall_acc_only": "Fall detection (SisFall, accelerometer only)",
              "deploy_har": "On-device HAR model (accelerometer-only 1D-CNN)", "export": "Edge deployment benchmarks"}
    for d, title in titles.items():
        f = ROOT / out_root / d / ("benchmark.md" if d == "export" else "results.md")
        if f.exists():
            parts.append(f"## {title}\n\n{f.read_text()}")
    out = ROOT / out_root / "RESULTS.md"
    out.write_text("\n".join(parts))
    print(f"\nWrote {out.relative_to(ROOT)}")


def serve(what, out_root):
    env = {**os.environ, "SAFESTEP_OUTPUTS": out_root}
    if what == "api":
        cmd = [sys.executable, "-m", "uvicorn", "app.api:app", "--port", "8000"]
        print("API docs: http://localhost:8000/docs")
    else:
        cmd = [sys.executable, "-m", "streamlit", "run", "app/dashboard.py"]
    subprocess.run(cmd, cwd=ROOT, env=env)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="synthetic data + tiny budgets (plumbing check)")
    p.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    p.add_argument("--serve", choices=["api", "dashboard"], help="start a server instead of training")
    args = p.parse_args()
    out_root = "outputs_smoke" if args.smoke else "outputs"
    data_dir = "data/synthetic" if args.smoke else "data/raw"

    if args.serve:
        return serve(args.serve, out_root)

    t0 = time.time()
    for stage in STAGES:
        if stage not in args.stages:
            continue
        print(f"\n{'=' * 70}\n== {stage}\n{'=' * 70}", flush=True)
        if stage == "data":
            stage_data(args.smoke)
        elif stage in CONFIGS:
            train(stage, args.smoke, out_root)
        elif stage == "export":
            stage_export(out_root, data_dir)
        elif stage == "demo":
            stage_demo(out_root, data_dir)
        elif stage == "report":
            stage_report(out_root, args.smoke)
    flag = " --smoke" if args.smoke else ""
    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min. Results: {out_root}/RESULTS.md\n"
          f"Dashboard: python run_all.py{flag} --serve dashboard\n"
          f"API:       python run_all.py{flag} --serve api")


if __name__ == "__main__":
    main()
