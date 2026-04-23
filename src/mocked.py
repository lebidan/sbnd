# A minimal SBND decoder model that you can use as a template for your own implementation.

import torch, torch.nn as nn
from torch import Tensor
from .codes import LinearCode
from .decoder import BaseDecoder
from .utils import get_rank_zero_logger

log = get_rank_zero_logger(__name__)


class MockedDecoder(BaseDecoder):

    def __init__(
        self,
        code: LinearCode,
        error_space: str = "codeword",
        compile: bool = False,
    ) -> None:
        super().__init__(code, error_space=error_space, compile=compile)

        log.info(f"Using the mocked decoder")

        # replace with your code
        # here we just use a single linear layer for demonstration purposes
        self.fc = nn.Linear(code.n + code.m, self.output_sz)

        # call last (compiles the forward graph once all submodules exist)
        self._maybe_compile()

    def forward(self, ym: Tensor, s: Tensor) -> Tensor:
        """Forward pass template"""
        # replace with your code
        x = torch.cat((ym, s), dim=1)
        return self.fc(x)


if __name__ == "__main__":
    pass
