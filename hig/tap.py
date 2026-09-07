"""Borrow the model output that diffusers throws away.

``callback_on_step_end`` fires after ``scheduler.step`` and carries only
latents, so the obvious way to reach the clean-image estimate is to difference
two consecutive latents. That works in float64 and is hopeless anywhere else:
the numerator subtracts near-equal quantities and amplifies whatever rounding
the *stored* latents already carry by a factor of ~20-30. Measured on the
guided steps, z0 comes back with a relative error of 7e-3 in fp16 and 6e-2 in
bf16 -- the dtypes SDXL and FLUX actually run in.

So instead of reconstructing the model output, we watch it go past. Wrapping
``scheduler.step`` for the duration of one pipeline call is a much smaller
intrusion than reimplementing the pipeline, and it makes z0 exact.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass

import torch

__all__ = ["StepRecord", "tap_scheduler"]


@dataclass
class StepRecord:
    """What the sampler fed into the most recent ``scheduler.step``."""

    sample: torch.Tensor | None = None
    model_output: torch.Tensor | None = None
    timestep: torch.Tensor | None = None

    @property
    def ready(self) -> bool:
        return self.sample is not None and self.model_output is not None


@contextmanager
def tap_scheduler(scheduler):
    """Record ``(sample, model_output, timestep)`` for each step, then restore.

    Only first-order solvers are supported: a multi-evaluation solver (Heun,
    DPM-Solver++ 2M) calls ``step`` more than once per iteration and the record
    would describe the last evaluation rather than the one that produced the
    latent the callback sees.
    """
    record = StepRecord()
    original = scheduler.step
    signature = inspect.signature(original)
    was_instance_attr = "step" in vars(scheduler)

    def step(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        record.sample = bound.arguments.get("sample")
        record.model_output = bound.arguments.get("model_output")
        record.timestep = bound.arguments.get("timestep")
        # Deliberately not recording the scheduler's own pred_original_sample:
        # pipelines call step(..., return_dict=False), which drops it, and a
        # field that is always None is worse than no field. z0_from_output is
        # pinned against it in the tests instead.
        return original(*args, **kwargs)

    scheduler.step = step
    try:
        yield record
    finally:
        if was_instance_attr:
            scheduler.step = original
        else:
            del scheduler.step
