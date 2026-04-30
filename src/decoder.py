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
        If True, `self.compile()` is invoked by `_maybe_compile()`. Subclasses
        MUST call `_maybe_compile()` at the end of their `__init__` — after
        every parameter, buffer, and submodule has been registered — so that
        the traced graph sees the fully-constructed module.

    Attributes set by the base class (do not override)
    --------------------------------------------------
    self.error_space         : str, as passed in.
    self.output_sz           : int, `code.n` if `error_space == "codeword"`,
                               else `code.k`. Use this to size the final
                               projection layer of the subclass.
    self.example_input_array : tuple[Tensor, Tensor], dummy `(ym, s)` inputs
                               used by Lightning for shape inference in the
                               model summary.
    """

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
        """Subclasses must call this LAST in their __init__ (after all params/buffers are registered)."""
        if self._compile:
            log.info("Compiling model forward for faster training")
            self.compile()

    @abstractmethod
    def forward(self, ym: Tensor, s: Tensor) -> Tensor: ...


if __name__ == "__main__":
    pass
