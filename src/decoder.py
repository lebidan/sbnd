# Shared abstract base class for SBND decoders.

import torch, torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from .codes import LinearCode
from .utils import get_rank_zero_logger

log = get_rank_zero_logger(__name__)


class BaseDecoder(nn.Module, ABC):
    """Abstract base class for syndrome-based neural decoders.

    A decoder consumes the channel-matched output `ym` of shape `(B, n)` and
    the bipolar syndrome `s` of shape `(B, m)`, and returns LLR-like logits
    for the predicted error pattern. The output shape is set by the
    `error_space` argument (see below). Subclasses must implement
    `forward(ym, s) -> Tensor`.

    Constructor arguments
    ---------------------
    code : LinearCode
        The code whose errors the decoder is trained to predict. Used to
        derive input/output sizes (`code.n`, `code.m`, `code.k`).
    error_space : {"codeword", "message"}, default "codeword"
        Space in which the target error vector lives, which fixes the decoder
        output size:
          - "codeword" → output shape `(B, n)`: predicts the full n-bit error
            pattern `e_cw = c_hat XOR c_true` in the codeword space.
          - "message"  → output shape `(B, k)`: predicts the k-bit error
            pattern `e_msg = (Ginv @ e_cw) mod 2` directly in the message
            space. Only meaningful when `code.Ginv` is available (it is, for
            every code loaded via `LinearCode`).
        MUST match the datamodule's `error_space` — a mismatch is caught at
        `trainer.fit` start by `SBNDLitModule.on_fit_start`. The value is
        stored on `self` so it survives checkpoint save/reload.
    compile : bool, default False
        If True, `self.compile()` is invoked by `_maybe_compile()`. Note that
        `_maybe_compile()` is NOT called from `__init__`: compilation is deferred
        and triggered by the training runtime (`SBNDLitModule.on_train_start`),
        deliberately AFTER Lightning's `ModelSummary` has run. The summary runs a
        forward under `FlopCounterMode`, and if that dispatch-mode trace hits an
        already-`torch.compile`d module it becomes the first graph dynamo traces
        and poisons its cache, making every subsequent training step substantially
        slower (observed as tens of percent). Compiling after the summary avoids
        this while keeping the full summary table (FLOPs and input/output sizes
        included). Consequences: a decoder
        used standalone (outside `SBNDLitModule`) stays eager until something
        calls `_maybe_compile()`; the compiled state does not survive checkpoint
        save/reload, so `sbnd-test` calls `_maybe_compile()` after loading.

    Attributes set by the base class (do not override)
    --------------------------------------------------
    self.error_space         : str, as passed in.
    self.output_sz           : int, `code.n` if `error_space == "codeword"`,
                               else `code.k`. Use this to size the final
                               projection layer of the subclass.
    self.example_input_array : tuple[Tensor, Tensor], dummy `(ym, s)` inputs
                               used by Lightning for shape inference in the
                               model summary.

    Eval-time iteration override (iterative decoders only)
    -------------------------------------------------------
    A decoder that loops one tied block opts in by setting the class attribute
    `_iters_attr` to the name of its iteration-count attribute. `sbnd-test` then
    calls `configure_eval(n_iters, code)`; while `self._eval_active` holds, the
    subclass's forward stacks the readout of every iteration into `(T, B, n)`
    logits and returns `self._select_iteration(logits, s)`.
    """

    # Class-level defaults: checkpoints pickle the whole decoder, so `__init__`
    # does not rerun on load and older pickles lack these instance attributes.
    _iters_attr: str | None = None  # iteration-count attribute; None = not iterative
    _trained_iters: int | None = None  # set by configure_eval; None = no override
    # float32 (n, m) parity-check matrix, registered by configure_eval. Annotation
    # only: a class-level value would make register_buffer refuse the name.
    eval_Ht: Tensor

    def __init__(
        self,
        code: LinearCode,
        error_space: str = "codeword",
        compile: bool = False,
    ) -> None:
        super().__init__()
        if error_space not in ("codeword", "message"):
            raise ValueError(
                f"error_space must be 'codeword' or 'message', got {error_space!r}"
            )
        self.error_space = error_space
        self.output_sz = code.k if error_space == "message" else code.n
        self._compile = compile
        self.example_input_array = (torch.zeros(1, code.n), torch.zeros(1, code.m))

    def _maybe_compile(self) -> None:
        """Compile the decoder's forward, once, if `compile=True`.

        Idempotent: safe to call multiple times (repeated fits, DDP ranks). This
        is intentionally NOT called from `__init__` — the training runtime calls
        it after the model summary has run; see the `compile` argument docstring
        for the (measured) reason.
        """
        # getattr: decoders pickled before `_compile` existed lack it. Test torch's own
        # state, not a flag of ours: a pickled flag would outlive the compiled forward.
        if getattr(self, "_compile", False) and self._compiled_call_impl is None:
            log.info("Compiling model forward")
            self.compile()

    def configure_eval(self, n_iters: int, code: LinearCode) -> None:
        """Evaluate at `n_iters` iterations, with syndrome-based early exit.

        Each frame outputs the readout of the first iteration whose hard decision
        matches the syndrome; frames never matching fall back to the readout at
        min(n_iters, trained count). `code` is the one the model was trained on.
        """
        name = type(self).__name__
        if self._iters_attr is None:
            raise ValueError(
                f"n_iters override is not supported by {name}: not an iterative decoder"
            )
        if self.error_space != "codeword":
            raise ValueError(
                "n_iters override needs error_space=codeword (the early exit checks "
                f"the syndrome), got {self.error_space!r}"
            )
        if n_iters < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}")
        if self._trained_iters is None:  # a second call must not overwrite it
            self._trained_iters = int(getattr(self, self._iters_attr))
        log.info(
            f"{name}: trained with {self._iters_attr}={self._trained_iters}, "
            f"evaluating with {n_iters} and syndrome early exit"
        )
        setattr(self, self._iters_attr, n_iters)
        device = next(self.parameters()).device
        self.register_buffer(
            "eval_Ht", code.Ht.to(device, torch.float32), persistent=False
        )

    @property
    def _eval_active(self) -> bool:
        return self._trained_iters is not None and not self.training

    def _select_iteration(self, logits: Tensor, s: Tensor) -> Tensor:
        """Pick one readout per frame out of `logits` (T, B, n), given bipolar `s` (B, m).

        First iteration whose hard decision (logit < 0 means bit in error) has
        syndrome `s`; else iteration min(T, trained count). Masked selection, no
        data-dependent control flow, so it compiles with fullgraph=True.
        """
        # GF(2) product in fp32: bf16 is only exact up to 256 bits per check
        with torch.autocast(logits.device.type, enabled=False):
            synd = 1 - 2 * (((logits < 0).float() @ self.eval_Ht) % 2)
        valid = (synd == s).all(dim=-1)  # (T, B)
        fallback = min(logits.shape[0], self._trained_iters or 0) - 1
        # argmax returns the first maximal index, i.e. the first valid iteration
        t = torch.where(valid.any(dim=0), valid.int().argmax(dim=0), fallback)
        return logits.take_along_dim(t[None, :, None], dim=0).squeeze(0)

    @abstractmethod
    def forward(self, ym: Tensor, s: Tensor) -> Tensor: ...


if __name__ == "__main__":
    pass
