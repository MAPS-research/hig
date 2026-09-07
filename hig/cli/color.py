"""``hig-color`` -- generate an image whose colour histogram matches a reference."""

from __future__ import annotations

import argparse
import sys

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hig-color",
        description="Generate an image constrained to a reference image's colour histogram.",
    )
    p.add_argument("--prompt", required=True, help="what to generate")
    p.add_argument("--reference", required=True, help="image supplying the target histogram")
    p.add_argument("--out", required=True, help="where to write the result")
    p.add_argument("--backbone", default="sdxl", choices=["sdxl", "flux"])
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--steps", type=int, default=None, help="denoising steps (backbone default)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--noise-levels",
        type=float,
        nargs="*",
        default=[0.65, 0.5, 0.35, 0.2],
        metavar="SIGMA",
        help="where in the denoising process to intervene, as a fraction of noise "
        "remaining. Portable across backbones, unlike step indices. Empty = "
        "post-hoc only.",
    )
    p.add_argument(
        "--guided-steps",
        type=int,
        nargs="*",
        default=None,
        metavar="I",
        help="step indices instead of noise levels; only meaningful for one schedule",
    )
    p.add_argument("--channels", default="rgb", help="colour channels the histogram covers")
    p.add_argument("--levels", type=int, default=16, help="quantisation levels per channel")
    p.add_argument(
        "--no-post-hoc",
        action="store_true",
        help="skip the final exact match; the histogram will be close but not exact",
    )
    p.add_argument("--save-baseline", metavar="PATH", help="also write the unguided image")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import torch
    from PIL import Image

    from hig.binning import GridBinning
    from hig.guidance import HistogramGuidance
    from hig.metrics import histogram_kl
    from hig.ot import HistogramMatcher
    from hig.pipelines import load_backbone

    res = args.resolution
    binning = GridBinning(args.channels, levels=args.levels)
    reference = np.array(Image.open(args.reference).convert("RGB").resize((res, res)))
    target = binning.bin_histogram(reference).astype(np.float64)

    pipe, adapter, defaults = load_backbone(args.backbone, device=args.device)
    if args.steps:
        defaults["num_inference_steps"] = args.steps

    def generate(guide=None):
        gen = torch.Generator(device=args.device).manual_seed(args.seed)
        kw = dict(prompt=args.prompt, height=res, width=res, generator=gen, **defaults)
        if guide is None:
            return np.array(pipe(**kw).images[0])
        with guide:
            return np.array(
                pipe(
                    **kw,
                    callback_on_step_end=guide,
                    callback_on_step_end_tensor_inputs=["latents"],
                ).images[0]
            )

    matcher = HistogramMatcher(binning, target, seed=args.seed)
    if args.guided_steps is not None:
        guide = HistogramGuidance(adapter, matcher, steps=args.guided_steps, height=res, width=res)
    elif args.noise_levels:
        guide = HistogramGuidance(
            adapter, matcher, noise_levels=args.noise_levels, height=res, width=res
        )
    else:
        guide = None

    image = generate(guide)
    if guide is not None:
        print(f"guided at steps {guide.guided_steps}", file=sys.stderr)
    print(f"HistKL before post-hoc: {histogram_kl(reference, image, binning):.4f}", file=sys.stderr)

    if not args.no_post_hoc:
        image = matcher(image)
        pixels = image.shape[0] * image.shape[1]  # not res*res: pipelines may round
        exact = np.array_equal(binning.bin_histogram(image), matcher.integer_target(pixels))
        print(
            f"HistKL after post-hoc : {histogram_kl(reference, image, binning):.4f} "
            f"(bin-for-bin exact: {exact})",
            file=sys.stderr,
        )

    Image.fromarray(image).save(args.out)
    if args.save_baseline:
        Image.fromarray(generate(None)).save(args.save_baseline)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
