"""Load any supported dataset as windows with the common schema."""


def load_windows(name, data_dir="data/raw", **kwargs):
    if name == "uci_har":
        from src.data.uci_har import load_uci_har
        return load_uci_har(data_dir)
    if name == "sisfall":
        from src.data.sisfall import load_sisfall_windows
        return load_sisfall_windows(data_dir, **kwargs)
    if name == "pamap2":
        from src.data.pamap2 import load_pamap2
        return load_pamap2(data_dir, **kwargs)
    raise ValueError(f"Unknown dataset {name}")
