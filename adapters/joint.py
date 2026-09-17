from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from adapters.base import BaseAdapter
from adapters.text import TextAdapter
from adapters.image import ImageAdapter
from utils.profiler import profile_columns

TEXT_DIM = 384
IMAGE_DIM = 512
PRESENCE_DIM = 3


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class JointAdapter(BaseAdapter):
    """
    Builds one fixed-dimension embedding per record from an optional subset
    of {tabular, text, image}, so the same Domain Classifier Test used for
    text/image can also catch correlation breaks between modalities that no
    single-modality check can see — a per-field/per-embedding marginal shift
    isn't required for a joint combination to become anomalous.

    Unlike TextAdapter/ImageAdapter (stateless, pretrained), tabular
    vectorization needs per-project baseline statistics, so this adapter
    also exposes fit_tabular_schema() as a /fit-time-only step, and
    transform() takes those stats as an extra argument (still ABC-compliant
    since Python's abstractmethod only checks the method is overridden, not
    its exact signature).

    KNOWN LIMITATION (proven by tests/test_joint_adapter.py's
    test_pure_correlation_inversion_is_a_known_blind_spot): a pure
    correlation inversion between two modalities — where each modality's
    own marginal distribution is completely unchanged and only the pairing
    between them flips — is NOT detected. drift/embedding_detector.py's
    LogisticRegression is a purely additive linear model over the
    concatenated vector this adapter produces, and separating such a swap
    requires an XOR-style interaction no additive linear model can
    represent, regardless of sample size or signal strength. This was
    found empirically and deliberately left unaddressed in this pass to
    avoid modifying the shared detector (which text/image also depend on)
    or adding new trainable components here. Correlation breaks that
    co-occur with any marginal movement in either modality, or with a
    shift in which modalities are typically present (via the presence
    mask), are unaffected by this limitation.
    """

    def fit_tabular_schema(self, records: List[dict]) -> Dict[str, Any]:
        """
        Profiles the tabular sub-fields extracted from records, reusing
        utils/profiler.py unchanged (exactly like the existing
        /fit/{project_id} handler), and computes the per-field baseline
        statistics needed to vectorize tabular data consistently between
        /fit and /analyze. Matches the existing tabular endpoint's policy:
        only fields classified as continuous or categorical are kept —
        ignored/uncertain fields are silently dropped, not guessed at.
        """
        tabular_records = [r.get("tabular") or {} for r in records]
        df = pd.DataFrame(tabular_records)

        stats: Dict[str, Any] = {}
        if df.empty or len(df.columns) == 0:
            return stats

        for p in profile_columns(df):
            field = p["name"]
            series = df[field].dropna()
            if len(series) == 0:
                continue

            if p["monitor"] is True:
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                if len(numeric) == 0:
                    continue
                stats[field] = {
                    "type": "continuous",
                    "mean": float(numeric.mean()),
                    "std": float(numeric.std()),
                }
            elif p["monitor"] == "Categorical":
                freq = series.astype(str).value_counts(normalize=True).to_dict()
                stats[field] = {"type": "categorical", "frequencies": freq}
            # False ("ignore") and "Review" fields contribute no dimension,
            # same as how /fit/{project_id} silently excludes them today.

        return stats

    def _tabular_subvector(self, tabular: Optional[dict], fields: List[str], tabular_stats: Dict[str, Any]) -> np.ndarray:
        vec = np.zeros(len(fields), dtype=np.float32)
        if not tabular:
            return vec

        for i, field in enumerate(fields):
            if field not in tabular or tabular[field] is None:
                continue
            spec = tabular_stats[field]
            value = tabular[field]
            if spec["type"] == "continuous":
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                std = spec["std"]
                vec[i] = (value - spec["mean"]) / std if std > 0 else 0.0
            else:  # categorical
                vec[i] = spec["frequencies"].get(str(value), 0.0)

        return vec

    def transform(self, raw_data: List[dict], tabular_stats: Dict[str, Any] = None) -> np.ndarray:
        """
        raw_data: list of {"tabular": dict|None, "text": str|None, "image": base64 str|None}.
        tabular_stats: the dict returned by fit_tabular_schema() at /fit time,
        reused unchanged at /analyze time so both sides encode consistently.
        """
        tabular_stats = tabular_stats or {}
        fields = sorted(tabular_stats.keys())
        f_dim = len(fields)
        total_dim = f_dim + TEXT_DIM + IMAGE_DIM + PRESENCE_DIM

        n = len(raw_data)
        if n == 0:
            return np.empty((0, total_dim), dtype=np.float32)

        has_tabular = [bool(r.get("tabular")) for r in raw_data]
        has_text = [bool(r.get("text")) for r in raw_data]
        has_image = [bool(r.get("image")) for r in raw_data]

        for i in range(n):
            if not (has_tabular[i] or has_text[i] or has_image[i]):
                raise ValueError(
                    f"Record at index {i} has no tabular, text, or image content — "
                    "at least one modality is required per record."
                )

        # Batch-embed only the records that actually have each modality —
        # much faster than per-record calls into the pretrained models, and
        # matches the existing text/image endpoints' batch-oriented calls.
        text_embeddings: Dict[int, np.ndarray] = {}
        text_indices = [i for i in range(n) if has_text[i]]
        if text_indices:
            embedded = _l2_normalize(TextAdapter().transform([raw_data[i]["text"] for i in text_indices]))
            text_embeddings = dict(zip(text_indices, embedded))

        image_embeddings: Dict[int, np.ndarray] = {}
        image_indices = [i for i in range(n) if has_image[i]]
        if image_indices:
            embedded = _l2_normalize(ImageAdapter().transform([raw_data[i]["image"] for i in image_indices]))
            image_embeddings = dict(zip(image_indices, embedded))

        out = np.zeros((n, total_dim), dtype=np.float32)
        for i, record in enumerate(raw_data):
            offset = 0
            if f_dim > 0:
                out[i, offset:offset + f_dim] = self._tabular_subvector(record.get("tabular"), fields, tabular_stats)
            offset += f_dim

            if i in text_embeddings:
                out[i, offset:offset + TEXT_DIM] = text_embeddings[i]
            offset += TEXT_DIM

            if i in image_embeddings:
                out[i, offset:offset + IMAGE_DIM] = image_embeddings[i]
            offset += IMAGE_DIM

            out[i, offset] = 1.0 if has_tabular[i] else 0.0
            out[i, offset + 1] = 1.0 if has_text[i] else 0.0
            out[i, offset + 2] = 1.0 if has_image[i] else 0.0

        return out
