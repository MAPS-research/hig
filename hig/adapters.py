"""Per-backbone glue: latent layout and the VAE round trip.

Everything else in HIG is backbone-agnostic. What differs between SDXL and the
FLUX family is only:

* how latents are laid out (4-D feature maps vs. sequences of 2x2 patch tokens),
* how latents are scaled (a factor, plus a shift for the FLUX VAE),
* how many channels the VAE has.

An adapter answers exactly that, in two methods: ``decode`` takes whatever the
pipeline hands around and returns an 8-bit image; ``encode`` goes back.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

__all__ = ["BackboneAdapter", "SDXLAdapter", "FluxAdapter", "adapter_for"]


class BackboneAdapter(ABC):
    """Latent layout and VAE round trip for one pipeline family.

    The adapter never mutates the pipeline. Promoting ``pipe.vae`` to float32
    in place looks appealing -- the round trip runs once per guided step and its
    error is the dominant failure mode of the method -- but the pipeline's own
    final decode hands the VAE latents in *its* dtype and diffusers only inserts
    a cast for the fp16-with-force_upcast case. Changing the VAE underneath it
    breaks generation. To run the VAE in float32, load it that way.
    """

    def __init__(self, pipe, vae=None):
        self.pipe = pipe
        #: The round trip may go through a different VAE than the pipeline's --
        #: a float32 copy, say. Passing one here keeps the pipeline untouched.
        self.vae = pipe.vae if vae is None else vae

    @property
    def vae_dtype(self) -> torch.dtype:
        return next(self.vae.parameters()).dtype

    # -- latent scaling, shared -------------------------------------------

    @property
    def scaling_factor(self) -> float:
        return float(self.vae.config.scaling_factor)

    @property
    def shift_factor(self) -> float:
        """Non-zero for the FLUX VAE, absent for SDXL's. Dropping it silently
        produces a plausible-looking but wrong image."""
        return float(getattr(self.vae.config, "shift_factor", None) or 0.0)

    # -- layout, per family -----------------------------------------------

    @abstractmethod
    def to_spatial(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Pipeline layout -> ``(B, C, H/8, W/8)``."""

    @abstractmethod
    def to_pipeline(self, latents: torch.Tensor) -> torch.Tensor:
        """``(B, C, H/8, W/8)`` -> pipeline layout."""

    # -- the round trip ----------------------------------------------------

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, height: int, width: int) -> np.ndarray:
        """Latents -> ``(H, W, 3) uint8``, the space the histogram lives in."""
        z = self.to_spatial(latents, height, width).float()
        z = (z / self.scaling_factor + self.shift_factor).to(self.vae_dtype)
        image = self.vae.decode(z, return_dict=False)[0]
        image = ((image.float() / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
        return image[0].permute(1, 2, 0).cpu().numpy()

    @torch.no_grad()
    def encode(self, image: np.ndarray, like: torch.Tensor) -> torch.Tensor:
        """``(H, W, 3) uint8`` -> latents in ``like``'s layout and dtype.

        Uses the posterior mode rather than a sample: the embedding application
        requires bit-reproducible behaviour, and sampling here would inject
        noise from outside the seeded generator.
        """
        x = torch.from_numpy(np.ascontiguousarray(image)).to(self.vae.device)
        x = (x.permute(2, 0, 1)[None].float() / 127.5 - 1.0).to(self.vae_dtype)
        z = self.vae.encode(x, return_dict=False)[0].mode().float()
        z = (z - self.shift_factor) * self.scaling_factor
        return self.to_pipeline(z).to(like.dtype)


class SDXLAdapter(BackboneAdapter):
    """SDXL hands 4-D latents straight through; nothing to unpack."""

    def to_spatial(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if latents.ndim != 4:
            raise ValueError(f"expected 4-D latents, got {tuple(latents.shape)}")
        return latents

    def to_pipeline(self, latents: torch.Tensor) -> torch.Tensor:
        return latents


class FluxAdapter(BackboneAdapter):
    """FLUX-family latents arrive as sequences of 2x2 patch tokens.

    ``(B, (H/16)*(W/16), 4C)`` has to become ``(B, C, H/8, W/8)`` before the VAE
    will look at it. diffusers implements this as a private static method whose
    signature has moved between releases, so HIG carries its own -- six lines,
    and a test pins it against the diffusers version.
    """

    def __init__(self, pipe, vae=None):
        super().__init__(pipe, vae)
        self.latent_channels = int(self.vae.config.latent_channels)
        # 8 for the FLUX VAE; patches then halve each spatial dimension again
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def to_spatial(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if latents.ndim == 4:  # already unpacked
            return latents
        batch, tokens, channels = latents.shape
        h = height // (self.vae_scale_factor * 2)
        w = width // (self.vae_scale_factor * 2)
        if h * w != tokens:
            raise ValueError(
                f"{tokens} tokens do not tile a {height}x{width} image "
                f"(expected {h * w}); height and width must be multiples of "
                f"{self.vae_scale_factor * 2}"
            )
        latents = latents.view(batch, h, w, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch, channels // 4, h * 2, w * 2)

    def to_pipeline(self, latents: torch.Tensor) -> torch.Tensor:
        batch, channels, h, w = latents.shape
        latents = latents.view(batch, channels, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(batch, (h // 2) * (w // 2), channels * 4)


def adapter_for(pipe, **kwargs) -> BackboneAdapter:
    """Pick an adapter from the pipeline's class name."""
    name = type(pipe).__name__
    if "Flux" in name:
        return FluxAdapter(pipe, **kwargs)
    if "StableDiffusionXL" in name:
        return SDXLAdapter(pipe, **kwargs)
    raise NotImplementedError(f"no adapter for {name}; add one by supplying to_spatial/to_pipeline")
