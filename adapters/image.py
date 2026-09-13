import base64
import io
from typing import List

import numpy as np

from adapters.base import BaseAdapter

_MODEL_NAME = "resnet18"
_model = None
_preprocess = None


def _get_model_and_preprocess():
    """
    Lazy singleton: resnet18 pretrained on ImageNet with the final
    classification layer stripped, used purely as a fixed 512-dim
    feature extractor for the Domain Classifier Test.
    """
    global _model, _preprocess
    if _model is None:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        model.fc = nn.Identity()
        model.eval()
        _model = model
        _preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _model, _preprocess


def _decode_image(raw):
    """
    Accepts a base64-encoded image string (JSON payload), raw bytes, or an
    already-decoded PIL Image, and normalizes it to RGB regardless of
    source mode (grayscale, RGBA, palette, etc.) or size.
    """
    from PIL import Image

    if isinstance(raw, str):
        raw = base64.b64decode(raw)
    if isinstance(raw, (bytes, bytearray)):
        image = Image.open(io.BytesIO(raw))
    else:
        image = raw
    return image.convert("RGB")


class ImageAdapter(BaseAdapter):
    """
    Converts a batch of images (base64-encoded strings) into resnet18
    penultimate-layer embeddings for the Domain Classifier Test. Handles
    common formats (JPEG, PNG) and mismatched sizes via resize/center-crop.
    """

    model_name = _MODEL_NAME

    def transform(self, raw_data: List[str]) -> np.ndarray:
        import torch

        if not raw_data:
            return np.empty((0, 512), dtype=np.float32)

        model, preprocess = _get_model_and_preprocess()

        tensors = []
        for item in raw_data:
            image = _decode_image(item)
            tensors.append(preprocess(image))

        batch = torch.stack(tensors)
        with torch.no_grad():
            features = model(batch)

        return features.numpy().astype(np.float32)
