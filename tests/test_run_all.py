import yaml

import run_all


def test_smoke_config_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all, "ROOT", run_all.ROOT)
    p = run_all.smoke_config("configs/fall.yaml", "outputs_smoke_test")
    cfg = yaml.safe_load((run_all.ROOT / p).read_text())
    assert cfg["data_dir"] == "data/synthetic"
    assert cfg["output_dir"].startswith("outputs_smoke_test/")
    assert cfg["train"]["epochs"] <= 10
    import shutil
    shutil.rmtree(run_all.ROOT / "outputs_smoke_test")


def test_every_config_maps_to_a_script():
    for stage, cfgs in run_all.CONFIGS.items():
        for c in cfgs:
            assert (run_all.ROOT / c).exists(), c
            assert run_all.script_for(c) in {"src.train_baseline", "src.train_deep", "src.train_fall"}
    assert run_all.script_for("configs/default.yaml") == "src.train_baseline"
    assert run_all.script_for("configs/fall_acc_only.yaml") == "src.train_fall"
    assert run_all.script_for("configs/deploy_har.yaml") == "src.train_deep"
