"""Loading a backbone with the pieces HIG expects.

Each family needs a specific scheduler and dtype, and SDXL additionally wants a
VAE that behaves in fp16. Getting any of it wrong fails quietly -- a
higher-order scheduler breaks the step tap, the stock SDXL VAE produces NaNs in
fp16 -- so the combinations live here rather than in every script.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from hig.adapters import BackboneAdapter, adapter_for

__all__ = ["Backbone", "BACKBONES", "load_backbone"]


@dataclass(frozen=True)
class Backbone:
    repo: str
    pipeline: str
    dtype: torch.dtype
    variant: str | None = None
    #: SDXL's own VAE overflows in fp16; this one is a drop-in that does not.
    vae_repo: str | None = None
    #: HIG reads the model output through scheduler.step, so the solver must
    #: evaluate the model once per step. Heun and DPM-Solver++ 2M do not.
    scheduler: str | None = None
    defaults: dict = field(default_factory=dict)


BACKBONES = {
    "sdxl": Backbone(
        repo="stabilityai/stable-diffusion-xl-base-1.0",
        pipeline="StableDiffusionXLPipeline",
        dtype=torch.float16,
        variant="fp16",
        vae_repo="madebyollin/sdxl-vae-fp16-fix",
        scheduler="DDIMScheduler",
        defaults={"num_inference_steps": 30, "guidance_scale": 7.5},
    ),
    "flux": Backbone(
        repo="black-forest-labs/FLUX.1-dev",
        pipeline="FluxPipeline",
        dtype=torch.bfloat16,
        defaults={"num_inference_steps": 30, "guidance_scale": 3.5},
    ),
}


def load_backbone(
    name: str, device: str = "cuda", **overrides
) -> tuple[object, BackboneAdapter, dict]:
    """Return ``(pipe, adapter, generation_defaults)`` for a named backbone."""
    import diffusers

    if name not in BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    spec = BACKBONES[name]

    kwargs = {"torch_dtype": spec.dtype}
    if spec.variant:
        kwargs["variant"] = spec.variant
    pipe = getattr(diffusers, spec.pipeline).from_pretrained(spec.repo, **kwargs).to(device)

    if spec.vae_repo:
        pipe.vae = diffusers.AutoencoderKL.from_pretrained(
            spec.vae_repo, torch_dtype=spec.dtype
        ).to(device)
    if spec.scheduler:
        pipe.scheduler = getattr(diffusers, spec.scheduler).from_pretrained(
            spec.repo, subfolder="scheduler"
        )
    pipe.set_progress_bar_config(disable=True)

    return pipe, adapter_for(pipe), {**spec.defaults, **overrides}
