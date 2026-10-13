from src.models.cnn import CNN1D
from src.models.cnn_lstm import CNNLSTM
from src.models.transformer import HARTransformer

REGISTRY = {"cnn1d": CNN1D, "cnn_lstm": CNNLSTM, "transformer": HARTransformer}


def build_model(name, n_channels, n_classes, seq_len, **params):
    if name not in REGISTRY:
        raise ValueError(f"Unknown model {name}, choose from {list(REGISTRY)}")
    if name == "transformer":
        params.setdefault("seq_len", seq_len)
    return REGISTRY[name](n_channels, n_classes, **params)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
