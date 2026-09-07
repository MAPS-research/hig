"""HIG -- Histogram-constrained Image Generation.

Training-free distributional control for diffusion models. A target histogram is
enforced by transporting the sampler's intermediate clean-image estimate onto it
at a few points during denoising, then once more, exactly, on the finished
image.

    from hig import GridBinning, HistogramGuidance, HistogramMatcher, load_backbone

    pipe, adapter, defaults = load_backbone("sdxl")
    binning = GridBinning("rgb", levels=16)
    matcher = HistogramMatcher(binning, target_weights, seed=0)

    with HistogramGuidance(adapter, matcher, noise_levels=[.65, .5, .35, .2]) as guide:
        image = pipe(prompt, height=1024, width=1024,
                     callback_on_step_end=guide,
                     callback_on_step_end_tensor_inputs=["latents"], **defaults).images[0]

    final = matcher(np.array(image))   # exact, bin for bin
"""

from hig.adapters import BackboneAdapter, FluxAdapter, SDXLAdapter, adapter_for
from hig.binning import Binning, GridBinning, MultiOptionBinning
from hig.codec import embedding_from_histogram, weights_from_embedding
from hig.guidance import HistogramGuidance
from hig.metrics import histogram_kl
from hig.ot import HistogramMatcher, solve_plan, to_counts
from hig.pipelines import BACKBONES, load_backbone
from hig.schedules import AffineSchedule, affine_schedule_from
from hig.tap import tap_scheduler

__version__ = "0.1.0"

__all__ = [
    "BACKBONES",
    "AffineSchedule",
    "BackboneAdapter",
    "Binning",
    "FluxAdapter",
    "GridBinning",
    "HistogramGuidance",
    "HistogramMatcher",
    "MultiOptionBinning",
    "SDXLAdapter",
    "adapter_for",
    "affine_schedule_from",
    "embedding_from_histogram",
    "histogram_kl",
    "load_backbone",
    "solve_plan",
    "tap_scheduler",
    "to_counts",
    "weights_from_embedding",
]
