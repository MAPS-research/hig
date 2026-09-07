"""Affine noise-schedule coefficients, the one thing HIG needs from a sampler.

Every first-order deterministic sampler in diffusers writes the latent at step
``i`` in the affine form

    x_i = a_i * z0 + b_i * eps

and holds ``(z0, eps)`` fixed across a single step. Given the model output for
that step -- which :mod:`hig.tap` snatches on the way past -- ``z0`` follows in
closed form, and re-noising a corrected ``z0`` collapses to a single term.

That is the whole model-specific surface HIG needs, which is why it can live
inside ``callback_on_step_end`` instead of reimplementing each pipeline.

Coefficients per family::

    DDIM / DDPM / DPM-Solver      a = sqrt(alpha_bar)   b = sqrt(1 - alpha_bar)
    Euler & friends (sigma-based) a = 1                 b = sigma
    flow matching (FLUX, SD3.5)   a = 1 - sigma         b = sigma
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["AffineSchedule", "affine_schedule_from"]


@dataclass(frozen=True)
class AffineSchedule:
    """``a`` and ``b`` for steps ``0 .. n`` (``n`` entries of each, plus the
    terminal state, so both tensors have length ``num_inference_steps + 1``)."""

    a: torch.Tensor
    b: torch.Tensor
    family: str
    prediction_type: str = "epsilon"

    def __post_init__(self) -> None:
        if self.a.shape != self.b.shape:
            raise ValueError(f"a{tuple(self.a.shape)} and b{tuple(self.b.shape)} must match")

    def __len__(self) -> int:
        return self.a.numel() - 1

    # -- the two operations HIG performs -----------------------------------

    def z0_from_output(
        self, x_cur: torch.Tensor, model_output: torch.Tensor, i: int
    ) -> torch.Tensor:
        """Clean-image estimate implied by the model output at step ``i``.

        Exact, and free of the cancellation that makes
        :meth:`recover_z0` unusable below float32.
        """
        work = torch.promote_types(x_cur.dtype, torch.float32)
        a0, b0 = (c.to(work) for c in self._coeffs(i, x_cur.device)[:2])
        x, out = x_cur.to(work), model_output.to(work)

        if self.family == "flow-matching":  # model predicts the velocity eps - z0
            z0 = x - b0 * out
        elif self.prediction_type == "epsilon":
            z0 = (x - b0 * out) / a0
        elif self.prediction_type == "v_prediction":
            z0 = a0 * x - b0 * out
        elif self.prediction_type == "sample":
            z0 = out
        else:
            raise NotImplementedError(f"prediction_type={self.prediction_type!r}")
        return z0.to(x_cur.dtype)

    def recover_z0(self, x_cur: torch.Tensor, x_next: torch.Tensor, i: int) -> torch.Tensor:
        """Same estimate, from two consecutive latents instead of the model output.

        Kept as a cross-check for the tests, not for production use: the
        denominator subtracts near-equal products, so the error already present
        in the stored latents comes back amplified ~20-30x. Fine in float64,
        useless in the fp16/bf16 the backbones actually run in. Prefer
        :meth:`z0_from_output`.
        """
        work = torch.promote_types(x_cur.dtype, torch.float32)
        a0, b0, a1, b1 = (c.to(work) for c in self._coeffs(i, x_cur.device))
        det = a0 * b1 - a1 * b0
        if det.abs() < 1e-8:
            raise ValueError(
                f"step {i}: schedule is degenerate (det={det.item():.3e}); "
                "z0 cannot be recovered from two latents here"
            )
        z0 = (b1 * x_cur.to(work) - b0 * x_next.to(work)) / det
        return z0.to(x_cur.dtype)

    def renoise(
        self, x_next: torch.Tensor, z0_old: torch.Tensor, z0_new: torch.Tensor, i: int
    ) -> torch.Tensor:
        """Put a corrected ``z0`` back on the trajectory, keeping ``eps`` fixed.

        Because ``x_{i+1} = a_{i+1} z0 + b_{i+1} eps`` and only ``z0`` changed,
        the update is exactly ``a_{i+1} * delta``.
        """
        work = torch.promote_types(x_next.dtype, torch.float32)
        a1 = self._coeffs(i, x_next.device)[2].to(work)
        delta = z0_new.to(work) - z0_old.to(work)
        return (x_next.to(work) + a1 * delta).to(x_next.dtype)

    # -- step selection ----------------------------------------------------

    def noise_fraction(self) -> torch.Tensor:
        """``b / (a + b)`` per step: a family-agnostic "how noisy is it here".

        Equals sigma exactly under flow matching, and is a monotone rescaling of
        it elsewhere, so a set of guidance points expressed in these units
        transfers between backbones. Step index does not -- the schedules put
        their steps in very different places.
        """
        return self.b / (self.a + self.b)

    def steps_at(self, levels) -> list[int]:
        """Step indices whose noise fraction is closest to each requested level."""
        nf = self.noise_fraction()[:-1]
        out = []
        for lvl in levels:
            idx = int(torch.argmin((nf - float(lvl)).abs()).item())
            if idx not in out:
                out.append(idx)
        return sorted(out)

    # -- internals ---------------------------------------------------------

    def _coeffs(self, i: int, device) -> tuple[torch.Tensor, ...]:
        if not 0 <= i < len(self):
            raise IndexError(f"step {i} out of range for a {len(self)}-step schedule")
        a, b = self.a.to(device), self.b.to(device)
        return a[i], b[i], a[i + 1], b[i + 1]


def affine_schedule_from(scheduler) -> AffineSchedule:
    """Read ``(a, b)`` off a diffusers scheduler that has already had
    ``set_timesteps`` called on it."""
    name = type(scheduler).__name__

    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")

    if "FlowMatch" in name:
        sigmas = _sigmas(scheduler, name)
        return AffineSchedule(
            a=1.0 - sigmas, b=sigmas, family="flow-matching", prediction_type="flow"
        )

    if getattr(scheduler, "sigmas", None) is not None:
        sigmas = _sigmas(scheduler, name)
        return AffineSchedule(
            a=torch.ones_like(sigmas), b=sigmas, family="sigma", prediction_type=prediction_type
        )

    if getattr(scheduler, "alphas_cumprod", None) is not None:
        timesteps = scheduler.timesteps
        if timesteps is None or len(timesteps) == 0:
            raise ValueError(f"{name}: call set_timesteps() first")
        ac = scheduler.alphas_cumprod.to(torch.float64).cpu()
        idx = timesteps.to(torch.long).cpu()
        # The step out of the last timestep lands on the scheduler's terminal
        # alpha_bar, not on another entry of `timesteps`.
        final = getattr(scheduler, "final_alpha_cumprod", None)
        if final is None:
            final = ac[0]
        alpha_bar = torch.cat([ac[idx], torch.as_tensor([final], dtype=torch.float64)])
        return AffineSchedule(
            a=alpha_bar.sqrt().float(),
            b=(1.0 - alpha_bar).sqrt().float(),
            family="alpha-bar",
            prediction_type=prediction_type,
        )

    raise NotImplementedError(f"{name}: no sigmas and no alphas_cumprod to read")


def _sigmas(scheduler, name: str) -> torch.Tensor:
    sigmas = scheduler.sigmas
    if sigmas is None or len(sigmas) == 0:
        raise ValueError(f"{name}: call set_timesteps() first")
    sigmas = sigmas.to(torch.float32).cpu()
    if len(sigmas) != len(scheduler.timesteps) + 1:
        raise ValueError(
            f"{name}: call set_timesteps() first -- expected len(sigmas) == "
            f"len(timesteps) + 1, got {len(sigmas)} and {len(scheduler.timesteps)}"
        )
    return sigmas
