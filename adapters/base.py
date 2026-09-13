from abc import ABC, abstractmethod
from typing import Any


class BaseAdapter(ABC):
    """
    Shared contract for all modality adapters (tabular, text, image).

    Each adapter converts a raw input batch into whatever representation its
    corresponding detector consumes: tabular passes columnar data through
    unchanged for KS/PSI/IQR; text/image return an (n_samples, embedding_dim)
    matrix for the Domain Classifier Test. This is what lets baseline storage,
    /analyze routing, and the dashboard treat all three modalities uniformly.
    """

    @abstractmethod
    def transform(self, raw_data: Any) -> Any:
        """Convert a raw batch into the representation its detector expects."""
        raise NotImplementedError
