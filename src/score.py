from __future__ import annotations

import json
import os
from typing import Dict, List

from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def init() -> None:
    global _model
    model_dir = os.environ.get("AZUREML_MODEL_DIR", "outputs/bge-m3-kr")
    _model = SentenceTransformer(model_dir)


def run(raw_data: str) -> Dict[str, List[List[float]]]:
    data = json.loads(raw_data)
    texts = data.get("texts", [])
    if not isinstance(texts, list):
        raise ValueError("'texts' must be a list of strings")

    assert _model is not None, "Model is not loaded"
    embeddings = _model.encode(texts, normalize_embeddings=True).tolist()
    return {"embeddings": embeddings}
