# Data augmentation transforms for SBND training, based on the code automorphisms

from importlib.metadata import requires

import torch, numpy as np
import h5py  # type: ignore[import-untyped]
from torch import Tensor
from scipy.io import loadmat  # type: ignore[import-untyped]
from .codes import LinearCode
from .utils import get_rank_zero_logger

log = get_rank_zero_logger(__name__)


class BCHPerms:
    def __init__(self, code: LinearCode, is_extended: bool = False) -> None:
        # precompute the subgroup of cyclic x frobenius permutations
        n = code.n - 1 if is_extended else code.n
        idx = torch.arange(n, dtype=torch.int64) # take_along_dim requires int64 indices
        b = torch.log2(torch.tensor(n + 1, dtype=torch.int)).int()
        perms = [(j + idx * 2**l) % n for j in range(n) for l in range(b)]
        perms = torch.stack(perms)
        self.n_perms = perms.size(0)
        if is_extended:
            # add the fixed extension bit to the end of each permutation
           perms = torch.hstack([perms, n * torch.ones(self.n_perms, 1, dtype=torch.int64)])
        self.perms = perms
        log.info(
            f"Data augmentation: using the {self.n_perms} cyclic x Frobenius permutations of {code}"
        )
        if is_extended:
            log.info(
                f"Note: This is only a small subset of the full automorphism group of the eBCH code"
            )

    def __call__(self, y: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        bs = y.size(0)
        perms = self.perms[torch.randint(self.n_perms, (bs,))]
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
        self.n_perms = self.perms.size(0)
        log.info(
            f"Data augmentation: using the {self.n_perms} quasi-cyclic permutations of code {code}"
        )

    def __call__(self, y: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        bs = y.size(0)
        perms = self.perms[torch.randint(self.n_perms, (bs,))]
        yp = y.take_along_dim(perms, dim=-1)
        ep = e.take_along_dim(perms, dim=-1)
        return yp, ep


class GenericPerms:
    def __init__(
        self, code: LinearCode, mat_file: str, num_perms: int | None = None
    ) -> None:
        if not mat_file.endswith(".mat"):
            mat_file += ".mat"
        try:
            # v7 mat files or earlier are supported by scipy.io
            matlab_data = loadmat(mat_file, squeeze_me=True)
        except NotImplementedError:
            # but not v7.3 (=HDF5) mat files, for which we need h5py
            with h5py.File(mat_file, "r") as f:
                if "perms" not in f:
                    raise ValueError(f"Dataset perms not found in {mat_file}")
                # take_along_dim requires int64 indices 
                perms = torch.from_numpy(f["perms"][:].astype(np.int64).transpose())
        else:
            if "perms" not in matlab_data:
                raise ValueError(f"Dataset perms not found in {mat_file}")
            # take_along_dim requires int64 indices 
            perms = torch.tensor(matlab_data["perms"], dtype=torch.int64)
        assert (
            perms.ndim == 2 and perms.size(1) == code.n
        ), "permutations must be a 2D array of shape (n_perms, code.n)"
        total_perms = perms.size(0)
        if num_perms is not None and num_perms > total_perms:
            log.warning(
                f"Requested {num_perms} permutations but file only contains {total_perms}; using all {total_perms}"
            )
        self.n_perms = total_perms if num_perms is None else min(num_perms, total_perms)
        self.perms = perms[: self.n_perms]
        log.info(
            f"Data augmentation: using {self.n_perms} permutations read from file {mat_file}"
        )

    def __call__(self, y: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        bs = y.size(0)
        perms = self.perms[torch.randint(self.n_perms, (bs,))]
        yp = y.take_along_dim(perms, dim=-1)
        ep = e.take_along_dim(perms, dim=-1)
        return yp, ep


if __name__ == "__main__":
    pass
