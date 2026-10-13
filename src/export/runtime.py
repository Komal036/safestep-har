"""Minimal on-device runtime: needs only numpy + onnxruntime (no PyTorch).

Normalization and softmax are baked into the exported graph, so the
device feeds raw sensor windows and gets class probabilities back.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def make_session(path, threads=1):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class OnnxClassifier:
    def __init__(self, model_path, meta_path=None, threads=1):
        self.model_path = str(model_path)
        self.session = make_session(model_path, threads)
        self.input_name = self.session.get_inputs()[0].name
        meta_path = meta_path or Path(model_path).with_name("meta.json")
        self.meta = json.loads(Path(meta_path).read_text()) if Path(meta_path).exists() else {}
        self.classes = self.meta.get("classes")
        self.threshold = self.meta.get("threshold")

    def predict_proba(self, windows):
        """windows: (n, T, C) or (T, C) raw float array -> (n, n_classes) probabilities."""
        x = np.asarray(windows, dtype=np.float32)
        if x.ndim == 2:
            x = x[None]
        return self.session.run(None, {self.input_name: x})[0]

    def predict(self, windows):
        return self.predict_proba(windows).argmax(1)
