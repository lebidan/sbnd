# A minimal SBND decoder model that you can use as a template for your own implementation.

import torch, torch.nn as nn
from torch import Tensor
from .codes import LinearCode
from .utils import get_rank_zero_logger

log = get_rank_zero_logger(__name__)


class MockedDecoder(nn.Module):

    def __init__(
        self,
        code: LinearCode,
        error_space: str = "codeword",
    ) -> None:
        super().__init__()

        # for model summary (keep this line)
        self.example_input_array = torch.zeros(1, code.n), torch.zeros(1, code.m)

        # model input/output sizes (you may want to keep these)
        self.error_space = error_space
        input_sz = code.n + code.m
        output_sz = code.k if error_space == "message" else code.n

        log.info(f"Using the mocked decoder")

        # replace with your code
        # here we just use a single linear layer for demonstration purposes
        self.fc = nn.Linear(input_sz, output_sz)

    def forward(self, ym: Tensor, s: Tensor) -> Tensor:
        """Forward pass template"""
        # replace with your code
        x = torch.cat((ym, s), dim=1)
        return self.fc(x)


if __name__ == "__main__":
    pass
