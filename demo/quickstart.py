"""Both applications, end to end, with no assets to download beyond the models.

    python demo/quickstart.py colour
    python demo/quickstart.py embed --text "your secret here"

The colour demo generates its own reference image, so nothing ships in the repo.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

RES = 1024
NOISE_LEVELS = [0.65, 0.5, 0.35, 0.2]


def generate(pipe, defaults, prompt, seed, guide=None, res=RES):
    kwargs = dict(
        prompt=prompt,
        height=res,
        width=res,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        **defaults,
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


def demo_colour(args):
    from hig.binning import GridBinning
    from hig.guidance import HistogramGuidance
    from hig.metrics import histogram_kl
    from hig.ot import HistogramMatcher
    from hig.pipelines import load_backbone

    pipe, adapter, defaults = load_backbone(args.backbone, device=args.device)
    binning = GridBinning("rgb", levels=16)

    print("generating the colour reference ...")
    reference = generate(pipe, defaults, args.reference_prompt, seed=7)
    Image.fromarray(reference).save("demo_reference.png")

    matcher = HistogramMatcher(binning, binning.bin_histogram(reference).astype(np.float64), seed=0)

    print("generating without guidance ...")
    baseline = generate(pipe, defaults, args.prompt, seed=42)
    Image.fromarray(baseline).save("demo_unguided.png")
    Image.fromarray(matcher(baseline)).save("demo_posthoc_only.png")

    print(f"generating with guidance at noise levels {NOISE_LEVELS} ...")
    guide = HistogramGuidance(adapter, matcher, noise_levels=NOISE_LEVELS, height=RES, width=RES)
    guided = generate(pipe, defaults, args.prompt, seed=42, guide=guide)
    final = matcher(guided)
    Image.fromarray(final).save("demo_guided.png")

    target = matcher.integer_target(RES * RES)
    print(f"\nguided at steps {guide.guided_steps}")
    print(f"HistKL unguided        : {histogram_kl(reference, baseline, binning):.4f}")
    print(f"HistKL after guidance  : {histogram_kl(reference, guided, binning):.4f}")
    print(f"HistKL after post-hoc  : {histogram_kl(reference, final, binning):.4f}")
    print(f"bin-for-bin exact      : {np.array_equal(binning.bin_histogram(final), target)}")
    print("\nCompare demo_posthoc_only.png against demo_guided.png: same histogram,")
    print("very different picture. That difference is what the guidance buys.")


def demo_embed(args):
    from hig.binning import GridBinning
    from hig.codec import embedding_from_histogram, weights_from_embedding
    from hig.embed import SoftPromptTuner
    from hig.guidance import HistogramGuidance
    from hig.ot import HistogramMatcher
    from hig.pipelines import load_backbone

    binning = GridBinning("rg", levels=64)  # 4096 bins; blue stays unconstrained
    tuner = SoftPromptTuner(device=args.device)
    if binning.num_bins != tuner.hidden_size:
        raise SystemExit(
            f"binning has {binning.num_bins} bins but the LM's hidden size is "
            f"{tuner.hidden_size}; they must match"
        )

    print(f"tuning a soft prompt for {tuner.token_count(args.text)} tokens ...")
    # Five initialisations at once; the group stops at the first that verifies.
    # Any single one is unreliable on a long text, and the extra rows are
    # nearly free at batch size one.
    result = tuner.fit(args.text, attempts=5, verbose=True)[0]
    print(f"  {result.steps} steps / {result.seconds:.0f}s | reproduces it: {result.verified}")
    if not result.verified:
        raise SystemExit("tuning never reproduced the text; raise --max-steps and retry")

    pipe, adapter, defaults = load_backbone(args.backbone, device=args.device)
    matcher = HistogramMatcher(binning, weights_from_embedding(result.embedding), seed=0)
    guide = HistogramGuidance(adapter, matcher, noise_levels=NOISE_LEVELS, height=RES, width=RES)

    print("generating the carrier image ...")
    carrier = matcher(generate(pipe, defaults, args.prompt, seed=42, guide=guide))
    Image.fromarray(carrier).save("demo_carrier.png")
    counts = binning.bin_histogram(carrier)
    print(f"histogram exact: {np.array_equal(counts, matcher.integer_target(RES * RES))}")

    print("reading it back out of demo_carrier.png ...")
    recovered = tuner.generate(embedding_from_histogram(counts), tuner.token_count(args.text) + 10)[
        0
    ]
    print(f"\nexact match: {args.text in recovered}")
    print(f"recovered  : {recovered[: len(args.text) + 40]!r}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--backbone", default="sdxl", choices=["sdxl", "flux"])
    p.add_argument("--device", default="cuda")
    sub = p.add_subparsers(dest="which", required=True)

    c = sub.add_parser("colour", aliases=["color"])
    c.add_argument("--prompt", default="an astronaut riding a horse on a rocky plain, 8k")
    c.add_argument("--reference-prompt", default="a sunset over the ocean, warm tones, 8k")
    c.set_defaults(run=demo_colour)

    e = sub.add_parser("embed")
    e.add_argument(
        "--text",
        default="The plan minimises movement, so the edit is the "
        "smallest one the constraint allows.",
    )
    e.add_argument("--prompt", default="a serene mountain lake at golden hour, 8k")
    e.set_defaults(run=demo_embed)

    args = p.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
