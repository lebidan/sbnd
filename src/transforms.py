# Data augmentation transforms for SBND training, based on the code automorphisms

import torch
from torch import Tensor
from .codes import LinearCode
from .utils import get_rank_zero_logger

log = get_rank_zero_logger(__name__)


class BCHPerms:
    def __init__(self, code: LinearCode) -> None:
        # precompute the subgroup of cyclic x frobenius permutations
        idx = torch.arange(code.n)
        b = torch.log2(torch.tensor(code.n + 1, dtype=torch.int)).int()
        perms = [(j + idx * 2**l) % code.n for j in range(code.n) for l in range(b)]
        self.perms = torch.stack(perms)
        log.info(
            f"Data augmentation: using the {self.perms.size(0)} standard BCH permutations of code {code}"
        )

    def __call__(self, y: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        n_perms, bs = self.perms.size(0), y.size(0)
        perms = self.perms[torch.randint(n_perms, (bs,))]
        yp = y.take_along_dim(perms, dim=-1)
        ep = e.take_along_dim(perms, dim=-1)
        return yp, ep


class QCPerms:
    def __init__(self, code: LinearCode, Zc: int = 1) -> None:
        assert (
            code.n // Zc
        ) * Zc == code.n, "the circulant size Zc must divide the code length"
        # precompute the Zc quasi-cyclic permutations
        perms = [
            Zc * (torch.arange(code.n) // Zc) + (torch.arange(code.n) + i) % Zc
            for i in range(Zc)
        ]
        self.perms = torch.stack(perms)
        log.info(
            f"Data augmentation: using the {self.perms.size(0)} QC permutations of code {code}"
        )

    def __call__(self, y: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        n_perms, bs = self.perms.size(0), y.size(0)
        perms = self.perms[torch.randint(n_perms, (bs,))]
        yp = y.take_along_dim(perms, dim=-1)
        ep = e.take_along_dim(perms, dim=-1)
        return yp, ep


if __name__ == "__main__":
    pass
