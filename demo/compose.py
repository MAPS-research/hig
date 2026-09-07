"""HIG composed with a LoRA and with a ControlNet.

    python demo/compose.py lora                      # SDXL + a style LoRA
    python demo/compose.py lora --backbone flux      # FLUX.1 + a style LoRA
    python demo/compose.py controlnet                # SDXL + Canny ControlNet
    python demo/compose.py controlnet --backbone flux

The guidance is a ``callback_on_step_end`` and knows nothing about the pipeline
that calls it, so adding a LoRA or swapping in a ControlNet pipeline changes
nothing on HIG's side. Each variant writes an unguided image and a guided one
that share the adapter, the seed and the prompt, and differ only in colour
distribution -- which is exactly what the histogram constraint pins.

Weights (all public):
  SDXL LoRA    TheLastBen/Papercut_SDXL               trigger "papercut"
  FLUX LoRA    alvdansen/frosting_lane_flux           trigger "frstingln illustration"
  SDXL CN      diffusers/controlnet-canny-sdxl-1.0    (needs opencv: uv sync --extra compose)
  FLUX CN      InstantX/FLUX.1-dev-Controlnet-Canny
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

RES = 1024
NOISE_LEVELS = [0.65, 0.5, 0.35, 0.2]
REFERENCE_PROMPT = "a sunset over the ocean, warm tones, 8k"

LORAS = {
    "sdxl": ("TheLastBen/Papercut_SDXL", "papercut.safetensors", "papercut, "),
    "flux": ("alvdansen/frosting_lane_flux", None, "frstingln illustration, "),
}
# repo, ControlNet class, pipeline class, load kwargs, name of the control-image argument
CONTROLNETS = {
    "sdxl": (
        "diffusers/controlnet-canny-sdxl-1.0",
        "ControlNetModel",
        "StableDiffusionXLControlNetPipeline",
        dict(torch_dtype=torch.float16, variant="fp16"),
        "image",
    ),
    "flux": (
        "InstantX/FLUX.1-dev-Controlnet-Canny",
        "FluxControlNetModel",
        "FluxControlNetPipeline",
        dict(torch_dtype=torch.bfloat16),
        "control_image",
    ),
}


def generate(pipe, defaults, prompt, seed, guide=None, **extra):
    kwargs = dict(
        prompt=prompt,
        height=RES,
        width=RES,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        **defaults,
        **extra,
    )
    if guide is None:
        return np.array(pipe(**kwargs).images[0])
    with guide:
        return np.array(
            pipe(
                **kwargs,
                callback_on_step_end=guide,
                callback_on_step_end_tensor_inputs=["latents"],
            ).images[0]
        )


def report(tag, reference, unguided, guided, final, matcher, binning, guide):
    from hig.metrics import histogram_kl

    move = np.abs(final.astype(np.float32) - guided.astype(np.float32)).mean()
    exact = np.array_equal(binning.bin_histogram(final), matcher.integer_target(final[..., 0].size))
    print(f"\n[{tag}] guided at steps {guide.guided_steps}")
    print(f"HistKL unguided        : {histogram_kl(reference, unguided, binning):.4f}")
    print(f"HistKL after guidance  : {histogram_kl(reference, guided, binning):.4f}")
    print(f"HistKL after post-hoc  : {histogram_kl(reference, final, binning):.4f}")
    print(f"post-hoc move          : {move:.2f} / 255")
    print(f"bin-for-bin exact      : {exact}")


def demo_lora(args):
    from hig.binning import GridBinning
    from hig.guidance import HistogramGuidance
    from hig.ot import HistogramMatcher
    from hig.pipelines import load_backbone

    pipe, adapter, defaults = load_backbone(args.backbone, device=args.device)
    binning = GridBinning("rgb", levels=16)

    print("generating the colour reference with the plain backbone ...")
    reference = generate(pipe, defaults, REFERENCE_PROMPT, seed=7)
    Image.fromarray(reference).save("compose_reference.png")
    matcher = HistogramMatcher(binning, binning.bin_histogram(reference).astype(np.float64), seed=0)

    repo, weight_name, trigger = LORAS[args.backbone]
    print(f"loading LoRA {repo} ...")
    pipe.load_lora_weights(repo, weight_name=weight_name)
    prompt = trigger + args.prompt

    print("LoRA, no guidance ...")
    unguided = generate(pipe, defaults, prompt, seed=42)
    Image.fromarray(unguided).save("compose_lora_unguided.png")

    print(f"LoRA + guidance at noise levels {NOISE_LEVELS} ...")
    guide = HistogramGuidance(adapter, matcher, noise_levels=NOISE_LEVELS, height=RES, width=RES)
    guided = generate(pipe, defaults, prompt, seed=42, guide=guide)
    final = matcher(guided)
    Image.fromarray(final).save("compose_lora_guided.png")

    report(f"lora/{args.backbone}", reference, unguided, guided, final, matcher, binning, guide)


def canny(image: np.ndarray, low: int = 100, high: int = 200) -> Image.Image:
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise SystemExit("the controlnet demo needs opencv: uv sync --extra compose") from e
    edges = cv2.Canny(image, low, high)
    return Image.fromarray(np.repeat(edges[..., None], 3, axis=-1))


def demo_controlnet(args):
    import diffusers

    from hig.adapters import adapter_for
    from hig.binning import GridBinning
    from hig.guidance import HistogramGuidance
    from hig.ot import HistogramMatcher
    from hig.pipelines import load_backbone

    repo, cn_cls, pipe_cls, load_kwargs, control_arg = CONTROLNETS[args.backbone]
    pipe, _, defaults = load_backbone(args.backbone, device=args.device)
    binning = GridBinning("rgb", levels=16)

    print("generating the colour reference and the structure image ...")
    reference = generate(pipe, defaults, REFERENCE_PROMPT, seed=7)
    Image.fromarray(reference).save("compose_reference.png")
    structure = generate(pipe, defaults, args.prompt, seed=3)
    Image.fromarray(structure).save("compose_structure.png")
    control = canny(structure)
    control.save("compose_canny.png")
    matcher = HistogramMatcher(binning, binning.bin_histogram(reference).astype(np.float64), seed=0)

    print(f"swapping in {repo} ...")
    controlnet = getattr(diffusers, cn_cls).from_pretrained(repo, **load_kwargs).to(args.device)
    # Same weights, same scheduler, a different forward pass. The adapter only
    # looks at the latent layout and the VAE, both of which are unchanged.
    pipe = getattr(diffusers, pipe_cls).from_pipe(pipe, controlnet=controlnet)
    pipe.set_progress_bar_config(disable=True)
    adapter = adapter_for(pipe)
    extra = {control_arg: control, "controlnet_conditioning_scale": args.conditioning_scale}

    print("ControlNet, no guidance ...")
    unguided = generate(pipe, defaults, args.prompt, seed=42, **extra)
    Image.fromarray(unguided).save("compose_controlnet_unguided.png")

    print(f"ControlNet + guidance at noise levels {NOISE_LEVELS} ...")
    guide = HistogramGuidance(adapter, matcher, noise_levels=NOISE_LEVELS, height=RES, width=RES)
    guided = generate(pipe, defaults, args.prompt, seed=42, guide=guide, **extra)
    final = matcher(guided)
    Image.fromarray(final).save("compose_controlnet_guided.png")

    report(
        f"controlnet/{args.backbone}", reference, unguided, guided, final, matcher, binning, guide
    )
    print("compose_canny.png is the structure both images follow; compose_reference.png")
    print("is the colour distribution the guided one carries.")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--backbone", default="sdxl", choices=["sdxl", "flux"])
    p.add_argument("--device", default="cuda")
    sub = p.add_subparsers(dest="which", required=True)

    lora = sub.add_parser("lora")
    lora.add_argument("--prompt", default="an astronaut riding a horse on a rocky plain")
    lora.set_defaults(run=demo_lora)

    cn = sub.add_parser("controlnet")
    cn.add_argument("--prompt", default="a lighthouse on a cliff above the sea, 8k")
    cn.add_argument("--conditioning-scale", type=float, default=0.5)
    cn.set_defaults(run=demo_controlnet)

    args = p.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
