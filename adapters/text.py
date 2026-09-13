from typing import List

import numpy as np

from adapters.base import BaseAdapter

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def _get_model():
    """Lazy singleton so the model is loaded once per process, not per request."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


class TextAdapter(BaseAdapter):
    """
    Converts a batch of raw strings into sentence embeddings for the
    Domain Classifier Test. Deterministic given the same model version;
    the underlying tokenizer truncates long strings to the model's max
    sequence length automatically, and empty strings embed without error.
    """

    model_name = _MODEL_NAME

    def transform(self, raw_data: List[str]) -> np.ndarray:
        texts = ["" if t is None else str(t) for t in raw_data]
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        model = _get_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)
