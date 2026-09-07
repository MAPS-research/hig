# Histogram-constrained Image Generation

**Exact, training-free distributional control for diffusion models via optimal transport** (ECCV 2026)

[Haoming Liu](https://hmdliu.site/), [Yuanhe Guo](https://ricercarg.github.io/), Yijia Cao, Shenji Wan, [Hongyi Wen](https://whongyi.github.io/)

[[Project page]](https://maps-research.github.io/hig/) · [[arXiv]](https://arxiv.org/abs/2606.31683) · [[PDF]](https://arxiv.org/pdf/2606.31683)

<p align="center">
  <img src="https://maps-research.github.io/hig/static/images/Overview.png" width="100%" alt="HIG overview">
</p>

**TL;DR.** Controllable generation spans a spectrum of granularity: text prompts and
LoRAs steer generation *globally*, while ControlNets anchor *local* structure. HIG fills
the middle ground by regulating the *distributional* properties of an image. Given any
target histogram over pixel colours, HIG drives the diffusion trajectory to match it
*exactly*, using a minimal-cost optimal transport plan applied as inference-time
guidance. It is training-free, interpretable, lightweight, and fully compatible with
existing controls.

This repository is a minimal, self-contained implementation covering two applications
on two backbones:

|                          | SDXL | FLUX.1 [dev] |
| ------------------------ | :--: | :----------: |
| Colour-constrained generation | ✓ | ✓ |
| Information embedding (Llama-3.1-8B) | ✓ | ✓ |

plus a demo of HIG composed with a LoRA and with a ControlNet on either backbone.

---

## Setup

Python ≥ 3.10 and one CUDA GPU (everything here was run on a single 80 GB A100 / H100).
Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/MAPS-research/hig-minimal.git && cd hig-minimal
uv sync                                   # core: colour control + information embedding
uv sync --extra compose                   # + peft / opencv for the LoRA & ControlNet demo
uv sync --extra metrics                   # + CLIP / aesthetic scorers (optional)
```

Model weights are fetched into the Hugging Face cache by repo id:

```bash
export HF_HOME=/path/with/space          # ~60 GB for everything
./scripts/download_models.sh             # sdxl flux llm metrics
./scripts/download_models.sh --only sdxl,llm
./scripts/download_models.sh --check     # what is already cached
```

FLUX.1 [dev] is gated: accept its licence on Hugging Face and run `hf auth login`
(or set `HF_TOKEN`) before downloading. `meta-llama/Llama-3.1-8B` is gated too, so the
script pulls the `NousResearch/Meta-Llama-3.1-8B` mirror by default (same weights);
set `LLAMA_REPO` to override.

All commands below assume the environment is active (`source .venv/bin/activate`) or are
prefixed with `uv run`.

---

## Usage

### Demos

```bash
python demo/quickstart.py colour                        # colour control, SDXL
python demo/quickstart.py colour --backbone flux
python demo/quickstart.py embed --text "the sentence to hide"
python demo/quickstart.py embed --backbone flux --text "the sentence to hide"

python demo/compose.py lora                             # HIG + a style LoRA
python demo/compose.py controlnet --backbone flux       # HIG + a Canny ControlNet
```

`quickstart.py colour` generates its own reference image, then writes the unguided
result, the reference histogram applied post-hoc only, and the guided result. The last
two share a histogram bin for bin and look very different; that difference is what the
guidance buys. `quickstart.py embed` tunes a soft prompt for the text, generates a
carrier image whose colour histogram encodes it, and reads the text back out of the PNG.

`compose.py` runs the same constraint on top of a public LoRA (Papercut for SDXL,
Frosting Lane for FLUX) or a Canny ControlNet (`diffusers/controlnet-canny-sdxl-1.0`,
`InstantX/FLUX.1-dev-Controlnet-Canny`). HIG is a `callback_on_step_end`, so nothing on
its side changes when an adapter is added.

### Command-line tools

```bash
# colour control: match the histogram of a reference image
hig-color --prompt "an astronaut riding a horse on a rocky plain" \
          --reference sunset.png --out astronaut.png \
          --backbone sdxl --save-baseline unguided.png

# information embedding: tune, encode, decode
hig-embed train  --text-file secret.txt --out prompt.npy
hig-embed encode --embedding prompt.npy --prompt "a serene mountain lake" --out carrier.png
hig-embed decode --image carrier.png --expect-file secret.txt
```

Useful flags on both tools: `--backbone {sdxl,flux}`, `--seed`, `--resolution`, and
`--noise-levels` (where in the denoising process to intervene, as fractions of noise
remaining; default `0.65 0.5 0.35 0.2`, empty means post-hoc only). `hig-embed encode`
and `decode` also accept `--binning multi-option`, a scattered-colour binning that
needs no guided steps; encode and decode must use the same one.

### Python API

```python
import numpy as np
from hig import GridBinning, HistogramGuidance, HistogramMatcher, load_backbone

pipe, adapter, defaults = load_backbone("sdxl")           # or "flux"
binning = GridBinning("rgb", levels=16)                    # 16^3 = 4096 colour bins
target  = binning.bin_histogram(reference_image).astype(np.float64)
matcher = HistogramMatcher(binning, target, seed=0)

guide = HistogramGuidance(adapter, matcher, noise_levels=[0.65, 0.5, 0.35, 0.2])
with guide:
    image = pipe(prompt="...", height=1024, width=1024,
                 callback_on_step_end=guide,
                 callback_on_step_end_tensor_inputs=["latents"], **defaults).images[0]

final = matcher(np.array(image))                           # post-hoc pass: exact, bin for bin
```

Any diffusers pipeline whose scheduler evaluates the model once per step works the same
way; `adapter_for(pipe)` picks the latent layout for the SDXL and FLUX families.

---

## Code layout

```
hig/
  guidance.py    HistogramGuidance: the callback_on_step_end that applies decode -> OT -> encode
  schedules.py   (a, b) coefficients per step, z0 recovery, re-noising, noise level -> step index
  tap.py         borrows the model output from scheduler.step without patching the pipeline
  adapters.py    per-backbone latent layout and the VAE round trip (SDXL, FLUX)
  pipelines.py   backbone presets: repo, dtype, VAE, scheduler, generation defaults
  binning.py     grid and multi-option colour bins, plus their transport costs
  ot.py          integer apportionment, the network-simplex OT solve, pixel reassignment
  codec.py       soft prompt <-> histogram (softmax one way, centred log the other)
  embed.py       soft-prompt tuning against Llama-3.1-8B and greedy read-back
  metrics.py     HistKL, CLIP score, aesthetic score
  cli/           hig-color and hig-embed
demo/
  quickstart.py  both applications end to end, no assets needed
  compose.py     HIG on top of a LoRA and a ControlNet
scripts/
  download_models.sh
```

---

## Acknowledgements

Built on [diffusers](https://github.com/huggingface/diffusers),
[transformers](https://github.com/huggingface/transformers) and
[POT](https://github.com/PythonOT/POT). The demo uses
[Papercut SDXL](https://huggingface.co/TheLastBen/Papercut_SDXL),
[Frosting Lane FLUX](https://huggingface.co/alvdansen/frosting_lane_flux),
[controlnet-canny-sdxl-1.0](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0) and
[FLUX.1-dev-Controlnet-Canny](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny).
This work is supported by NYU Shanghai Center for Data Science and in part through the NYU IT High Performance Computing resources, services, and staff expertise.

## Licence

The code is released under the MIT licence (see `LICENSE`). Model weights keep their own terms: FLUX.1 [dev] is non-commercial, Llama-3.1 is under the Llama 3.1 Community License, and each third-party LoRA and ControlNet is governed by the licence on its Hugging Face page.

## Citation

```bibtex
@article{liu2026histogram,
  title={Histogram-constrained Image Generation},
  author={Liu, Haoming and Guo, Yuanhe and Cao, Yijia and Wan, Shenji and Wen, Hongyi},
  journal={arXiv preprint arXiv:2606.31683},
  year={2026}
}
```
