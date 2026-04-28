# Test-time scaling strategies for SBND models evaluation
#
# Each strategy wraps the decoding step inside the test loop and returns a
# binary error-pattern prediction (in the model's error_space, i.e. shape
# (bs, n) for codeword space, (bs, k) for message space). 
# 
# `SingleShotDecoder` is the no-TTS baseline (one forward pass). 
# 
# `SelfBoostingDecoder` is a sequential test-time scaling variant: the model
# iterates over its own predictions.
# 
# `TTADecoder` is a parallel test-time scaling variant: the model is run
# under multiple code automorphisms in parallel and their predictions are
# combined.
# 
# All three variants share the same `decode/validate/name/suffix` protocol 

from typing import Callable

import torch
from torch import Tensor

from .codes import LinearCode
from .model import SBNDLitModule
from .utils import get_rank_zero_logger

log = get_rank_zero_logger(__name__)


def bipolar_to_bit(x: Tensor) -> Tensor:
    return (x < 0).to(torch.int8)


class SingleShotDecoder:
    """No-TTS baseline: a single forward pass through the model.

    This is *not* a test-time scaling variant — it's the reference against
    which TTS strategies are compared. It exists in this module only so the
    test loop has a uniform `decode(...)` interface across the no-TTS and TTS
    code paths.
    """

    name: str = "no-tts"
    suffix: str = ""

    def validate(self, model: SBNDLitModule, code: LinearCode) -> None:
        return

    def decode(
        self,
        model: SBNDLitModule,
        code: LinearCode,
        ym: Tensor,
        syndromes: Tensor,
        targets: Tensor,
    ) -> Tensor:
        # `targets` is unused (only TTA needs it); accepted for protocol uniformity.
        return bipolar_to_bit(model(ym, syndromes))


class SelfBoostingDecoder:
    """Self-boosting (sequential TTS): the model iterates over its own predictions.

    First call: run the model on (ym, s_chan) to get an initial error-pattern
    prediction. Then check whether the prediction's syndrome matches the
    channel syndrome; for the samples where it does not, repeat the process.

    Only valid for models trained with error_space=codeword (the syndrome
    check requires predictions in codeword space).
    """

    name: str = "self-boosting"

    def __init__(self, num_iters: int = 5) -> None:
        if num_iters < 1:
            raise ValueError(f"num_iters must be >= 1 (got {num_iters})")
        self.num_iters = num_iters

    @property
    def suffix(self) -> str:
        return f"-sb{self.num_iters}"

    def validate(self, model: SBNDLitModule, code: LinearCode) -> None:
        es = getattr(model.decoder, "error_space", "codeword")
        if es != "codeword":
            raise ValueError(
                "SelfBoostingDecoder requires a model trained with error_space=codeword "
                f"(got error_space={es!r})"
            )

    def decode(
        self,
        model: SBNDLitModule,
        code: LinearCode,
        ym: Tensor,
        syndromes: Tensor,
        targets: Tensor,
    ) -> Tensor:
        # `targets` is unused (only TTA needs it); accepted for interface uniformity.
        Ht = code.Ht.to(device=ym.device, dtype=torch.float32)
        chan_synd = syndromes

        # iteration 0: feed the channel syndrome directly
        preds = bipolar_to_bit(model(ym, chan_synd))
        out_synd = 1 - 2 * ((preds.float() @ Ht) % 2)
        needs_update = torch.any(out_synd * chan_synd < 0, dim=-1)

        # subsequent iterations: feed the accumulated syndrome (residual):
        # chan_synd XOR (cumulative preds)·Hᵀ (in bipolar: chan_synd * out_synd)
        it = 1
        while torch.any(needs_update) and it < self.num_iters:
            residual = chan_synd[needs_update] * out_synd[needs_update]
            new_preds = bipolar_to_bit(model(ym[needs_update], residual))
            preds[needs_update] = (preds[needs_update] + new_preds) % 2
            out_synd = 1 - 2 * ((preds.float() @ Ht) % 2)
            needs_update = torch.any(out_synd * chan_synd < 0, dim=-1)
            it += 1
        return preds


class TTADecoder:
    """Test-time augmentation (parallel TTS): decode under multiple code automorphisms.

    For each test sample, draw `num_perms` random code automorphisms σ from
    the provided permutation generator (e.g. `BCHPerms`, `QCPerms`,
    `GenericPerms`). Apply each σ to the received word, recompute the
    syndrome of the permuted hard decisions, run the model, and inverse-permute
    the resulting logits back into the original coordinate ordering. Stop a
    sample as soon as one of its perms yields a prediction whose syndrome
    matches the channel syndrome; for samples that never pass the syndrome
    check, average the logits across all `num_perms` runs and threshold.

    Only valid for models trained with error_space=codeword (the syndrome
    check requires predictions in codeword space).
    """

    name: str = "tta"

    def __init__(self, transform: Callable, num_perms: int = 4) -> None:
        if num_perms < 1:
            raise ValueError(f"num_perms must be >= 1 (got {num_perms})")
        self._transform_factory = transform
        self.num_perms = num_perms
        self._transform: object | None = None

    @property
    def suffix(self) -> str:
        return f"-tta{self.num_perms}"

    def validate(self, model: SBNDLitModule, code: LinearCode) -> None:
        es = getattr(model.decoder, "error_space", "codeword")
        if es != "codeword":
            raise ValueError(
                "TTADecoder requires a model trained with error_space=codeword "
                f"(got error_space={es!r})"
            )
        # Instantiate the permutation generator now that we have the code.
        self._transform = self._transform_factory(code)
        n_avail = self._transform.n_perms  # type: ignore[attr-defined]
        if self.num_perms > n_avail:
            log.warning(
                f"TTADecoder: requested num_perms={self.num_perms} but only "
                f"{n_avail} permutations are available; sampling will repeat."
            )

    def _sample_perms(self, n: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Draw n random permutations and their inverses, both shape (n, code.n)."""
        assert self._transform is not None, "validate() must be called before decode()"
        perms_pool: Tensor = self._transform.perms  # type: ignore[attr-defined]
        n_pool: int = self._transform.n_perms  # type: ignore[attr-defined]
        idx = torch.randint(n_pool, (n,))
        perms = perms_pool[idx].to(device)
        perms_inv = perms.argsort(dim=-1)
        return perms, perms_inv

    def decode(
        self,
        model: SBNDLitModule,
        code: LinearCode,
        ym: Tensor,
        syndromes: Tensor,
        targets: Tensor,
    ) -> Tensor:
        Ht = code.Ht.to(device=ym.device, dtype=torch.float32)
        chan_synd = syndromes
        # `targets` (= channel error pattern e) is used purely as a binary vector
        # whose syndrome under permutation σ matches σ(z)·Hᵀ — i.e. as a stand-in
        # for the hard decisions, since e·Hᵀ ≡ z·Hᵀ (mod codewords). It is never
        # consulted as ground truth.
        e = targets.to(device=ym.device)
        bs, n = ym.shape

        # Sample bs * num_perms perms once and reshape to (bs, num_perms, n) so
        # each sample gets its own independent set of num_perms permutations.
        all_perms, all_inv = self._sample_perms(bs * self.num_perms, ym.device)
        all_perms = all_perms.reshape(bs, self.num_perms, n)
        all_inv = all_inv.reshape(bs, self.num_perms, n)

        # Per-perm logits in the *original* coordinate system (after inverse-perm).
        logits = torch.empty(
            (self.num_perms, bs, n), device=ym.device, dtype=torch.float32
        )

        # Permutation 0: run on every sample.
        ymp = ym.take_along_dim(all_perms[:, 0], dim=-1)
        ep = e.take_along_dim(all_perms[:, 0], dim=-1)
        in_synd = 1 - 2 * ((ep.float() @ Ht) % 2)
        logits[0] = model(ymp, in_synd).take_along_dim(all_inv[:, 0], dim=-1)
        out_synd = 1 - 2 * (((logits[0] < 0).float() @ Ht) % 2)
        needs_update = torch.any(out_synd * chan_synd < 0, dim=-1)

        # Subsequent permutations: only re-decode the samples that failed so far.
        # Successful samples carry their previous logits forward via `logits[p] = logits[p-1]`.
        for p in range(1, self.num_perms):
            logits[p] = logits[p - 1]
            if not torch.any(needs_update):
                break
            u = needs_update
            ymp = ym[u].take_along_dim(all_perms[u, p], dim=-1)
            ep = e[u].take_along_dim(all_perms[u, p], dim=-1)
            in_synd = 1 - 2 * ((ep.float() @ Ht) % 2)
            new_logits = model(ymp, in_synd).take_along_dim(all_inv[u, p], dim=-1)
            logits[p, u] = new_logits
            out_synd = 1 - 2 * (((logits[p] < 0).float() @ Ht) % 2)
            needs_update = torch.any(out_synd * chan_synd < 0, dim=-1)

        # For samples that never passed the syndrome check, average across all perms.
        if torch.any(needs_update):
            logits[-1, needs_update] = logits[:, needs_update].mean(dim=0)

        return bipolar_to_bit(logits[-1])


if __name__ == "__main__":
    pass
