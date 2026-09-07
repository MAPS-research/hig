"""The guidance hook: one callback, every backbone.

At a chosen step the sampler's clean-image estimate is decoded, transported onto
the target histogram, re-encoded, and put back on the trajectory::

    z0' = VAE_encode( match( VAE_decode(z0) ) )
    x_{i+1} <- x_{i+1} + a_{i+1} * (z0' - z0)

Because the transport is minimal, the correction is small, and the remaining
denoising steps refine away what artefacts it does leave. Guiding only near the
end -- or only after sampling has finished -- removes that refinement budget,
which is why the guided steps are chosen by noise level rather than by index.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from hig.adapters import BackboneAdapter
from hig.ot import HistogramMatcher
from hig.schedules import AffineSchedule, affine_schedule_from
from hig.tap import tap_scheduler

__all__ = ["HistogramGuidance"]


class HistogramGuidance:
    """A ``callback_on_step_end`` that pulls the histogram towards ``matcher``'s target.

    Used as a context manager, because it needs the scheduler tap in place for
    the duration of the pipeline call::

        with HistogramGuidance(adapter, matcher, noise_levels=[.65, .5, .35, .2]) as guide:
            image = pipe(prompt, height=h, width=w, latents=x_T,
                         callback_on_step_end=guide,
                         callback_on_step_end_tensor_inputs=["latents"]).images[0]

    Exactly one of ``steps`` and ``noise_levels`` may be given. ``noise_levels``
    is the portable one: it names *where in the denoising process* to intervene,
    which transfers across schedules; a step index does not, since DDIM and
    flow matching put their steps in very different places.
    """

    def __init__(
        self,
        adapter: BackboneAdapter,
        matcher: HistogramMatcher,
        *,
        steps: Sequence[int] | None = None,
        noise_levels: Sequence[float] | None = None,
        height: int | None = None,
        width: int | None = None,
    ):
        if (steps is None) == (noise_levels is None):
            raise ValueError("pass exactly one of steps= or noise_levels=")
        self.adapter = adapter
        self.matcher = matcher
        self.height = height
        self.width = width
        self._requested_steps = None if steps is None else sorted(set(steps))
        self._noise_levels = noise_levels

        self._tap = None
        self._record = None
        self._schedule: AffineSchedule | None = None
        #: steps actually guided, filled in once the schedule is known
        self.guided_steps: list[int] = []

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> HistogramGuidance:
        self._tap = tap_scheduler(self.adapter.pipe.scheduler)
        self._record = self._tap.__enter__()
        self._schedule = None
        self.guided_steps = []
        return self

    def __exit__(self, *exc) -> None:
        tap, self._tap, self._record = self._tap, None, None
        return tap.__exit__(*exc)

    # -- the callback ------------------------------------------------------

    def __call__(self, pipe, i: int, t, callback_kwargs: dict) -> dict:
        if self._record is None:
            raise RuntimeError("use HistogramGuidance as a context manager: `with guidance:`")

        if self._schedule is None:
            self._schedule = affine_schedule_from(pipe.scheduler)
            self.guided_steps = self._resolve_steps(self._schedule)

        if i not in self.guided_steps:
            return callback_kwargs

        latents = callback_kwargs["latents"]
        height, width = self._resolve_size(pipe, latents)

        z0 = self._schedule.z0_from_output(self._record.sample, self._record.model_output, i)
        image = self.adapter.decode(z0, height, width)
        matched = self.matcher(image)
        z0_matched = self.adapter.encode(matched, like=z0)

        callback_kwargs["latents"] = self._schedule.renoise(latents, z0, z0_matched, i)
        return callback_kwargs

    # -- helpers -----------------------------------------------------------

    def _resolve_steps(self, schedule: AffineSchedule) -> list[int]:
        if self._requested_steps is not None:
            n = len(schedule)
            bad = [s for s in self._requested_steps if not 0 <= s < n]
            if bad:
                raise ValueError(f"steps {bad} outside a {n}-step schedule")
            return list(self._requested_steps)
        return schedule.steps_at(self._noise_levels)

    def _resolve_size(self, pipe, latents: torch.Tensor) -> tuple[int, int]:
        if self.height is not None and self.width is not None:
            return self.height, self.width
        if latents.ndim == 4:  # spatial layout: read it off directly
            scale = getattr(pipe, "vae_scale_factor", 8)
            return latents.shape[-2] * scale, latents.shape[-1] * scale
        raise ValueError(
            "packed latents carry no image size; pass height= and width= to HistogramGuidance"
        )

    def post_hoc(self, image: np.ndarray) -> np.ndarray:
        """Final exact match on the 8-bit image.

        The during-sampling passes get the image close with the model still able
        to clean up after them; this last one is what makes the histogram exact,
        and it is the only step the embedding application cannot skip.
        """
        return self.matcher(image)
