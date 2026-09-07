"""``hig-embed`` -- hide text in an image's colour histogram, and read it back."""

from __future__ import annotations

import argparse
import sys

import numpy as np

DEFAULT_LLM = "NousResearch/Meta-Llama-3.1-8B"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hig-embed", description="Store text in an image's colour histogram."
    )
    p.add_argument("--llm", default=DEFAULT_LLM, help="hidden size must equal the bin count")
    p.add_argument("--device", default="cuda")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="tune a soft prompt that reproduces a text")
    t.add_argument("--text-file", required=True)
    t.add_argument("--out", required=True, help="destination .npy for the soft prompt")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--max-steps", type=int, default=3000)
    t.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="tune this many initialisations of the text at once and keep "
        "whichever works. Any single one is unreliable, and the group stops as "
        "soon as one of them succeeds, so the redundancy is nearly free",
    )
    t.add_argument("--verbose", action="store_true")

    e = sub.add_parser("encode", help="generate an image carrying a soft prompt")
    e.add_argument("--embedding", required=True)
    e.add_argument("--prompt", required=True, help="what the image should depict")
    e.add_argument("--out", required=True)
    e.add_argument("--backbone", default="sdxl", choices=["sdxl", "flux"])
    e.add_argument("--resolution", type=int, default=1024)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument(
        "--noise-levels",
        type=float,
        nargs="*",
        default=None,
        metavar="SIGMA",
        help="where to intervene, as a fraction of noise remaining. Default: "
        "0.65 0.5 0.35 0.2 for the grid, none (post-hoc only) for multi-option. "
        "Empty = post-hoc only",
    )

    d = sub.add_parser("decode", help="recover the text from an image")
    d.add_argument("--image", required=True)
    d.add_argument("--expect-file", help="text to compare against, for an exact-match report")

    for sp in (e, d):
        sp.add_argument("--channels", default="rg", help="two channels leave the third free")
        sp.add_argument("--levels", type=int, default=64, help="channels**levels must equal d")
        sp.add_argument(
            "--binning",
            default="grid",
            choices=["grid", "multi-option"],
            help="grid: one colour cube per bin, guided steps help. multi-option: "
            "each bin is a scattered set of colours, post-hoc alone suffices",
        )
        sp.add_argument(
            "--bins", type=int, default=4096, help="multi-option only: bin count (= hidden size)"
        )
        sp.add_argument(
            "--binning-seed",
            type=int,
            default=0,
            help="multi-option only: seeds the colour-to-bin permutation; encode and "
            "decode must agree on it",
        )
    return p


def _binning(args):
    from hig.binning import GridBinning, MultiOptionBinning

    if args.binning == "multi-option":
        return MultiOptionBinning(args.channels, num_bins=args.bins, seed=args.binning_seed)
    return GridBinning(args.channels, levels=args.levels)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "train":
        from hig.embed import SoftPromptTuner

        text = open(args.text_file).read()
        tuner = SoftPromptTuner(args.llm, device=args.device, max_steps=args.max_steps)
        result = tuner.fit(text, seed=args.seed, attempts=args.attempts, verbose=args.verbose)[0]
        np.save(args.out, result.embedding)
        print(
            f"{result.steps} steps / {result.seconds:.0f}s | final loss {result.loss:.5f} | "
            f"reproduces the text: {result.verified}",
            file=sys.stderr,
        )
        if not result.verified:
            print(
                "warning: greedy decoding never matched; the payload will not survive",
                file=sys.stderr,
            )
        print(args.out)
        return 0 if result.verified else 1

    if args.command == "encode":
        import torch
        from PIL import Image

        from hig.codec import weights_from_embedding
        from hig.guidance import HistogramGuidance
        from hig.ot import HistogramMatcher
        from hig.pipelines import load_backbone

        binning = _binning(args)
        embedding = np.load(args.embedding)
        if embedding.size != binning.num_bins:
            raise SystemExit(
                f"soft prompt has {embedding.size} dimensions but the binning has "
                f"{binning.num_bins} bins; they must match"
            )
        matcher = HistogramMatcher(binning, weights_from_embedding(embedding), seed=args.seed)
        pipe, adapter, defaults = load_backbone(args.backbone, device=args.device)
        res = args.resolution
        noise_levels = args.noise_levels
        if noise_levels is None:
            # A multi-option bin holds scattered colours, so the post-hoc pass
            # alone barely moves anything -- and its LP is tens of seconds a
            # solve at 1024^2, so guided steps would cost a lot for nothing.
            noise_levels = [] if args.binning == "multi-option" else [0.65, 0.5, 0.35, 0.2]
        kw = dict(
            prompt=args.prompt,
            height=res,
            width=res,
            generator=torch.Generator(device=args.device).manual_seed(args.seed),
            **defaults,
        )
        if noise_levels:
            guide = HistogramGuidance(
                adapter, matcher, noise_levels=noise_levels, height=res, width=res
            )
            with guide:
                image = np.array(
                    pipe(
                        **kw,
                        callback_on_step_end=guide,
                        callback_on_step_end_tensor_inputs=["latents"],
                    ).images[0]
                )
            print(f"guided at steps {guide.guided_steps}", file=sys.stderr)
        else:
            image = np.array(pipe(**kw).images[0])
        # The post-hoc pass is not optional here: an approximate histogram
        # decodes to an approximate vector, which decodes to the wrong tokens.
        image = matcher(image)
        pixels = image.shape[0] * image.shape[1]  # not res*res: pipelines may round
        exact = np.array_equal(binning.bin_histogram(image), matcher.integer_target(pixels))
        print(f"histogram exact: {exact}", file=sys.stderr)
        if not exact:
            raise SystemExit("post-hoc transport did not reach the target exactly")
        Image.fromarray(image).save(args.out)
        print(args.out)
        return 0

    if args.command == "decode":
        from PIL import Image

        from hig.codec import embedding_from_histogram
        from hig.embed import SoftPromptTuner

        binning = _binning(args)
        image = np.array(Image.open(args.image).convert("RGB"))
        embedding = embedding_from_histogram(binning.bin_histogram(image))
        tuner = SoftPromptTuner(args.llm, device=args.device)
        if args.expect_file:
            expected = open(args.expect_file).read()
            ok = tuner.reproduces(embedding, expected)[0]
            print(f"exact match: {ok}", file=sys.stderr)
            print(tuner.generate(embedding, tuner.token_count(expected) + 10)[0])
            return 0 if ok else 1
        print(tuner.generate(embedding, 512)[0])
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
