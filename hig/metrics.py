"""Evaluation: how well the constraint was met, and what it cost.

``histogram_kl`` needs nothing beyond numpy. The perceptual scores pull in CLIP
and live behind the ``metrics`` extra, since a user who only wants to generate
images should not have to install them.
"""

from __future__ import annotations

import numpy as np

from hig.binning import Binning, GridBinning

__all__ = ["histogram_kl", "CLIPScore", "AestheticScore"]

_EPS = 1e-10


def histogram_kl(reference: np.ndarray, image: np.ndarray, binning: Binning | None = None) -> float:
    """``KL(reference || image)`` over colour bins.

    Defaults to the rgb x16 grid the colour results are reported on. Both
    histograms are smoothed by 1e-10 before normalising, because an empty bin in
    the generated image would otherwise send the divergence to infinity.
    """
    binning = binning or GridBinning("rgb", levels=16)
    p = binning.bin_histogram(reference).astype(np.float64) + _EPS
    q = binning.bin_histogram(image).astype(np.float64) + _EPS
    p, q = p / p.sum(), q / q.sum()
    return float((p * np.log(p / q)).sum())


class CLIPScore:
    """Prompt adherence: ``100 * cos(image, text)``, the usual convention."""

    def __init__(self, model_id: str = "openai/clip-vit-large-patch14", device: str = "cuda"):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()

    def __call__(self, images: list[np.ndarray], prompts: list[str]) -> np.ndarray:
        if len(images) != len(prompts):
            raise ValueError(f"{len(images)} images but {len(prompts)} prompts")
        from PIL import Image

        pil = [Image.fromarray(im).resize((224, 224)) for im in images]
        inputs = self.processor(
            text=prompts, images=pil, return_tensors="pt", padding=True, truncation=True
        )
        with self.torch.no_grad():
            out = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
        return self.torch.diag(out.logits_per_text).cpu().numpy()


class AestheticScore:
    """LAION aesthetic predictor: a linear head on l2-normalised CLIP features."""

    URL = "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth"

    def __init__(self, weights: str = "assets/sa_0_4_vit_l_14_linear.pth", device: str = "cuda"):
        import os

        import open_clip
        import torch

        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"{weights} not found. Fetch it with scripts/download_models.sh --only metrics, "
                f"or download {self.URL} yourself."
            )
        self.torch = torch
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.model = self.model.to(device).eval()
        self.head = torch.nn.Linear(768, 1)
        self.head.load_state_dict(torch.load(weights, map_location="cpu"))
        self.head = self.head.to(device).eval()

    def __call__(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        from PIL import Image

        torch = self.torch
        batch = torch.cat([self.preprocess(Image.fromarray(im)).unsqueeze(0) for im in images])
        scores = []
        with torch.no_grad():
            for i in range(0, len(batch), batch_size):
                feats = self.model.encode_image(batch[i : i + batch_size].to(self.device))
                feats = feats / feats.norm(dim=-1, keepdim=True)
                scores.append(self.head(feats.float()).squeeze(-1).cpu())
        return torch.cat(scores).numpy()
