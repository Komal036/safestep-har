"""Download public HAR datasets into data/raw/.

Usage:
    python data/download.py --dataset uci_har
"""
import argparse
import io
import zipfile
from pathlib import Path

import requests

DATASETS = {
    "uci_har": "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "pamap2": "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
}


def _download_nested(name: str, folder: str, inner_zip: str, out_dir: Path) -> Path:
    target = out_dir / folder
    if target.exists():
        print(f"Already present: {target}")
        return target
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} (this can take a few minutes)...")
    resp = requests.get(DATASETS[name], timeout=600)
    resp.raise_for_status()
    outer = zipfile.ZipFile(io.BytesIO(resp.content))
    # UCI archives wrap the real dataset in a nested zip.
    inner = next((n for n in outer.namelist() if n.endswith(inner_zip)), None)
    if inner:
        zipfile.ZipFile(io.BytesIO(outer.read(inner))).extractall(out_dir)
    else:
        outer.extractall(out_dir)
    print(f"Extracted to {target}")
    return target


def download_uci_har(out_dir: Path) -> Path:
    return _download_nested("uci_har", "UCI HAR Dataset", "UCI HAR Dataset.zip", out_dir)


def download_pamap2(out_dir: Path) -> Path:
    return _download_nested("pamap2", "PAMAP2_Dataset", "PAMAP2_Dataset.zip", out_dir)


def sisfall_instructions(out_dir: Path):
    target = out_dir / "SisFall_dataset"
    if target.exists() and any(target.glob("S*/*.txt")):
        print(f"SisFall found at {target}")
        return
    print(
        "SisFall needs a manual download (no stable direct link).\n"
        "  1. Get the SisFall dataset (Sucerquia et al., 2017). Mirrors exist on\n"
        "     Kaggle and figshare; search 'SisFall dataset'.\n"
        f"  2. Extract it so files look like {target}/SA01/D01_SA01_R01.txt\n"
        "  3. Re-run this command to verify."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS) + ["sisfall"], required=True)
    p.add_argument("--out", default="data/raw")
    args = p.parse_args()
    if args.dataset == "uci_har":
        download_uci_har(Path(args.out))
    elif args.dataset == "pamap2":
        download_pamap2(Path(args.out))
    elif args.dataset == "sisfall":
        sisfall_instructions(Path(args.out))


if __name__ == "__main__":
    main()
