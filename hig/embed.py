"""Soft-prompt tuning and greedy read-back for the information-embedding application.

A single vector is optimised until the language model, given nothing but that
vector, greedily reproduces the target text. Two details carry the method:

* **Fixed norm.** The vector is reprojected to a constant norm after every step.
  It bounds what the histogram has to represent, and it pulls the vector towards
  the centred member of its shift class -- the only member :mod:`hig.codec` can
  recover, since a softmax forgets the shift.
* **Verify by decoding, not by loss.** A low cross-entropy does not guarantee
  that greedy decoding reproduces every token, so the stopping rule runs the
  actual decode. That is expensive relative to a training step, so it fires only
  when the loss sets a new best and the previous check has gone stale.

The soft prompt is spliced in through ``inputs_embeds`` rather than through a
PEFT adapter. A prompt-tuning adapter holds one vector and broadcasts it across
the batch, which makes it impossible to tune or decode several *different*
payloads at once -- and at batch size one an 8B model is bound by the cost of
reading its own weights, so batching is very nearly free throughput.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np

__all__ = ["SoftPrompt", "SoftPromptTuner"]


@dataclass
class SoftPrompt:
    """A tuned prompt and what it cost to find.

    ``steps`` is per payload -- a sample stops being optimised once it verifies.
    ``seconds`` is the whole batch's wall clock, the same for every member,
    because they were tuned together; divide by the batch size for a per-payload
    figure.
    """

    embedding: np.ndarray
    steps: int
    seconds: float
    loss: float
    verified: bool


class SoftPromptTuner:
    """A causal LM driven by one trainable vector in front of the sequence."""

    def __init__(
        self,
        model_id: str = "NousResearch/Meta-Llama-3.1-8B",
        device: str = "cuda",
        norm: float = 40.0,
        lr: float = 1e-1,
        max_steps: int = 3000,
        loss_threshold: float = 0.015,
        patience: int = 100,
        schedule: str = "cosine",
        min_lr_factor: float = 0.02,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.norm, self.lr = norm, lr
        self.max_steps, self.loss_threshold, self.patience = max_steps, loss_threshold, patience
        self.schedule, self.min_lr_factor = schedule, min_lr_factor

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embeddings = self.model.get_input_embeddings()
        self.hidden_size = int(self.embeddings.weight.shape[1])
        # Decoding starts from the same prefix training used. Naming the token in
        # a string would be Llama-specific: Llama's tokenizer adds its BOS on its
        # own, so a literal "<|begin_of_text|>" produces it twice, and some LMs have
        # no BOS at all, so the same literal arrives as seven ordinary tokens.
        self._prefix = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id else []

    def token_count(self, text: str) -> int:
        return int(self.tokenizer(text, return_tensors="pt").input_ids.shape[1])

    # -- splicing ----------------------------------------------------------

    def _prepend(self, soft, ids):
        """``(B, H)`` soft prompts in front of ``(B, L)`` token ids."""
        tokens = self.embeddings(ids).to(soft.dtype)
        return self.torch.cat([soft[:, None, :], tokens], dim=1)

    # -- read-back ---------------------------------------------------------

    def generate(
        self, embeddings, max_new_tokens: int, batch_size: int = 8, stopping_criteria=None
    ) -> list[str]:
        """Greedy continuation from each soft prompt.

        Decoding is bound by weight bandwidth rather than by attention over the
        prefix, so several payloads decode for very nearly the price of one.
        """
        torch = self.torch
        soft = self._as_batch(embeddings)
        if stopping_criteria is not None and batch_size < len(soft):
            # The criteria below are built for one specific batch of targets;
            # splitting the batch would silently compare against the wrong ones.
            raise ValueError(
                f"stopping_criteria are tied to a whole batch, but batch_size="
                f"{batch_size} would split {len(soft)} prompts"
            )
        out = []
        with torch.no_grad():
            for start in range(0, len(soft), batch_size):
                chunk = soft[start : start + batch_size]
                prefix = torch.tensor(
                    [self._prefix] * len(chunk), dtype=torch.long, device=self.device
                )
                embeds = self._prepend(chunk, prefix)
                mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
                ids = self.model.generate(
                    inputs_embeds=embeds,
                    attention_mask=mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                )
                out.extend(self.tokenizer.batch_decode(ids, skip_special_tokens=True))
        return out

    def reproduces(self, embeddings, texts, slack: int = 10, early_exit: bool = True) -> list[bool]:
        """Does each soft prompt greedily reproduce its text?

        With ``early_exit`` the batch stops once every sample has left its
        target's prefix. Greedy decoding is deterministic, so a payload that has
        gone wrong will not come back, and abandoning it early is most of the
        decode saved -- which matters because verification runs repeatedly while
        tuning is still converging, i.e. while it mostly fails.
        """
        texts = [texts] if isinstance(texts, str) else list(texts)
        longest = max(self.token_count(t) for t in texts) + slack
        criteria = self._prefix_criteria(texts) if early_exit else None
        outputs = self.generate(
            embeddings, longest, batch_size=len(texts), stopping_criteria=criteria
        )
        return [t in o for t, o in zip(outputs, texts, strict=True)]

    def _prefix_criteria(self, texts, check_every: int = 16, trim: int = 4):
        """Stop a sample once its output is no longer a prefix of its target.

        Checked on a stride, because decoding the batch back to text is not
        free, and the last few characters are dropped before comparing: a
        half-emitted word is not yet a divergence.
        """
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        tokenizer = self.tokenizer

        class LeftThePrefix(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                batch = input_ids.shape[0]
                blank = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
                if input_ids.shape[1] % check_every or batch != len(texts):
                    return blank
                decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                gone = [
                    not target.startswith(seen[: max(0, len(seen) - trim)])
                    for target, seen in zip(texts, decoded, strict=True)
                ]
                return torch.tensor(gone, dtype=torch.bool, device=input_ids.device)

        return StoppingCriteriaList([LeftThePrefix()])

    # -- tuning ------------------------------------------------------------

    def fit(
        self, texts, seed: int = 0, attempts: int = 1, verbose: bool = False
    ) -> list[SoftPrompt]:
        """Tune one soft prompt per text, all at once.

        Returns a list even for a single text, so callers do not branch on the
        shape of what they asked for.

        ``attempts`` tunes that many independent initialisations of every text
        and keeps whichever one works. Success from any single initialisation is
        variable, and the whole group stops the moment one of its members
        reproduces its text -- so redundancy is far cheaper than it sounds. At
        512 tokens the slowest of three initialisations took 2204 steps where
        the fastest took 353, and only the 353 has to be paid for.

        Within a group the decode check goes to the lowest-loss member that is
        due for one, because that check is a full greedy decode and loss is a
        free way to rank who deserves it. Loss only ranks, though: it cannot
        decide. Cross-entropy is a mean over hundreds of tokens and exact
        reproduction is all-or-nothing across them, so one token losing its
        argmax moves the mean by about as much as every token being slightly
        blunter. Only the decode can tell those apart.

        Batch size is bounded by the logits, not by the model: the loss needs a
        float32 copy of ``(batch, tokens, vocab)``, which is about 260 MB per row
        at 512 tokens against Llama's vocabulary. Note that ``attempts``
        multiplies the row count.
        """
        torch = self.torch
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        if attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {attempts}")

        group = np.repeat(np.arange(len(texts)), attempts)  # row -> which text
        ids, labels = self._training_batch([texts[g] for g in group])
        rows_total = len(group)
        offset = len(self._prefix)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        soft = torch.randn(rows_total, self.hidden_size, generator=generator).to(self.device)
        soft = torch.nn.Parameter(self._project(soft.to(torch.float32)))

        optimiser = torch.optim.AdamW([soft], lr=self.lr)
        scheduler = self._scheduler(optimiser)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

        started = time.time()
        solved = np.zeros(len(texts), dtype=bool)  # a text with a member that verified
        winner = np.full(len(texts), -1)  # which row of the group that was
        best = torch.full((rows_total,), float("inf"))
        losses = torch.full((rows_total,), float("inf"))
        last_check = np.full(rows_total, -self.patience)
        frozen = soft.detach().clone()
        per_text_steps = np.zeros(len(texts), dtype=int)
        counts = (labels != -100).sum(dim=1).clamp(min=1)

        for step in range(self.max_steps):
            # Only rows whose text is still unsolved go through the model.
            # Keeping a finished group in the batch would pay for a full forward
            # and backward to no purpose, and groups finish far apart.
            awake = np.flatnonzero(~solved[group])
            if not len(awake):
                break
            rows = torch.from_numpy(awake).to(self.device)

            embeds = self._prepend(soft[rows].to(torch.bfloat16), ids[rows])
            # An all-ones mask is correct despite the padding: it sits at the end
            # of each row, attention is causal, so no real token can see it -- and
            # the positions it occupies are dropped from the loss by their -100
            # labels.
            logits = self.model(
                inputs_embeds=embeds,
                attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long, device=self.device),
            ).logits
            # Position j of [soft] + ids predicts ids[j], so the target for
            # ids[k] is logits[k]. The prefix is fed at generation time and is
            # not a target, hence the offset -- and when there is no prefix the
            # soft prompt's own position must predict the first token, which is
            # exactly the position a hard-coded slice of 1 would have dropped.
            live_labels = labels[rows]
            span = logits[:, offset : offset + live_labels.shape[1]]
            token_loss = loss_fn(
                span.reshape(-1, logits.shape[-1]).float(), live_labels.reshape(-1)
            ).view(live_labels.shape)
            per_row = token_loss.sum(dim=1) / counts[rows]
            per_row.sum().backward()
            optimiser.step()
            optimiser.zero_grad()
            scheduler.step()
            with torch.no_grad():
                soft.data = self._project(soft.data)

            # `losses` persists, so a row whose group has finished keeps the loss
            # it stopped at rather than being overwritten by nothing.
            losses[torch.from_numpy(awake)] = per_row.detach().float().cpu()
            per_text_steps[~solved] = step + 1
            if verbose and step % 50 == 0:
                # stderr, so a caller whose stdout carries a result stays clean
                print(f"  step {step:5d}  loss {losses.tolist()}", file=sys.stderr)

            improved = losses < best
            best = torch.minimum(best, losses)
            due = [
                i
                for i in awake
                if improved[i]
                and best[i] <= self.loss_threshold
                and step - last_check[i] >= self.patience
            ]
            # one decode per group per step: the cheapest ranking of who has
            # earned the expensive check
            pick: dict[int, int] = {}
            for i in due:
                g = int(group[i])
                if g not in pick or losses[i] < losses[pick[g]]:
                    pick[g] = int(i)
            candidates = sorted(pick.values())
            if candidates:
                for i in candidates:
                    last_check[i] = step
                sub = soft.detach()[candidates].float().cpu().numpy()
                verdicts = self.reproduces(sub, [texts[group[i]] for i in candidates])
                for i, ok in zip(candidates, verdicts, strict=True):
                    if ok:
                        g = int(group[i])
                        solved[g], winner[g] = True, i
                        frozen[i] = soft.detach()[i]
                if solved.all():
                    break

        elapsed = time.time() - started
        vectors = soft.detach().float().cpu().numpy()
        pinned = frozen.float().cpu().numpy()
        out = []
        for t in range(len(texts)):
            members = np.flatnonzero(group == t)
            if winner[t] >= 0:
                # the member that reproduced its text, pinned at the step it did
                row, embedding = int(winner[t]), pinned[int(winner[t])]
            else:
                # nothing verified, so hand back the closest attempt rather than
                # whichever one happens to be first
                row = int(members[int(np.argmin(losses[members].numpy()))])
                embedding = vectors[row]
            out.append(
                SoftPrompt(
                    embedding=embedding,
                    steps=int(per_text_steps[t]),
                    seconds=elapsed,
                    loss=float(losses[row]),
                    verified=bool(winner[t] >= 0),
                )
            )
        return out

    # -- internals ---------------------------------------------------------

    def _as_batch(self, embeddings):
        arr = np.asarray(embeddings, dtype=np.float32)
        arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
        if arr.shape[1] != self.hidden_size:
            raise ValueError(
                f"soft prompts have {arr.shape[1]} dimensions but the model's hidden "
                f"size is {self.hidden_size}"
            )
        return self.torch.tensor(arr, device=self.device, dtype=self.torch.bfloat16)

    def _training_batch(self, texts):
        """Right-padded ids and labels; padding is masked out of the loss.

        Each row is ``[bos?] + text + eos * 3``, assembled from ids rather than
        from a string so that no tokenizer's special tokens are assumed. The
        three trailing ``eos`` are what teaches the prompt where the text ends.
        """
        torch = self.torch
        eos = [self.tokenizer.eos_token_id] * 3
        sequences = [
            torch.tensor(
                self._prefix
                + self.tokenizer(t, add_special_tokens=False, return_tensors=None).input_ids
                + eos,
                dtype=torch.long,
            )
            for t in texts
        ]
        width = max(len(s) for s in sequences)
        offset = len(self._prefix)
        pad = self.tokenizer.eos_token_id
        ids = torch.full((len(sequences), width), pad, dtype=torch.long)
        labels = torch.full((len(sequences), width - offset), -100, dtype=torch.long)
        for i, seq in enumerate(sequences):
            ids[i, : len(seq)] = seq
            labels[i, : len(seq) - offset] = seq[offset:]
        return ids.to(self.device), labels.to(self.device)

    def _project(self, soft):
        return soft / soft.norm(dim=-1, keepdim=True) * self.norm

    def _scheduler(self, optimiser):
        """Cosine by default.

        The original step decay halves the rate every 200 steps regardless of
        the budget, so past roughly 1500 steps there is no learning rate left
        and the remaining budget is spent doing nothing. Cosine ties the decay
        to ``max_steps`` instead, which matters for long payloads: they are the
        ones that need the extra steps.
        """
        torch = self.torch
        if self.schedule == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=self.max_steps, eta_min=self.lr * self.min_lr_factor
            )
        if self.schedule == "step":  # the original, kept for comparison
            return torch.optim.lr_scheduler.StepLR(optimiser, step_size=200, gamma=0.5)
        raise ValueError(f"unknown schedule {self.schedule!r}")
