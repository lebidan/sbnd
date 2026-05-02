# Code-automorphism transforms for SBND.
#
# These classes are used both for:
#   - training-time data augmentation, by passing one as the `transform`
#     argument of `SBNDDataModule` (only with pre-computed datasets — on-demand
#     mode generates fresh samples per step so it does not need augmentation);
#   - test-time augmentation, by passing one as the `transform` argument of
#     `sbnd.tts.TTADecoder`.
#
# Common interface
# ----------------
# Each class exposes:
#   self.perms        : Tensor of shape (n_perms, code.n), int64.
#                       Each row is a valid permutation of [0, ..., code.n - 1]
#                       that is a code automorphism.
#   self.n_perms      : int, number of permutations stored in `self.perms`.
#   self.synd_maps    : Tensor of shape (n_perms, m, m), float32. Per-perm
#                       linear map taking synd(z) → synd(σ(z)). Populated
#                       lazily by `compute_synd_maps()` on the first call to
#                       `sample_perms` to avoid wasting uncessary compute
#                       when training. Used by `TTADecoder`.
#   __call__(y, e)    : draw one random permutation per sample and apply it to
#                       `y` and `e` jointly. Used by the training-augmentation
#                       path.
#   sample_perms(bs)  : draw `bs` random permutations and return them together
#                       with their inverses and the matching syndrome maps.
#                       Used by `TTADecoder`. Implemented once in the
#                       `BasePerms` base class.

import torch, numpy as np
import h5py  # type: ignore[import-untyped]
from torch import Tensor
from scipy.io import loadmat  # type: ignore[import-untyped]
from .codes import LinearCode
from .utils import get_rank_zero_logger

log = get_rank_zero_logger(__name__)


def _syndrome_basis(H: Tensor) -> Tensor:
    """Return Z (m, n) over GF(2) such that Z @ H.T = I_m mod 2.

    Each row Z[i] is a vector in GF(2)^n with syndrome e_i (the i-th standard
    basis vector). Built once per code via Gauss-Jordan on [H | I_m]; used by
    `BasePerms.compute_synd_maps` to lift permutations from coordinate space
    to syndrome space (so TTA can derive synd(σ(z)) directly from synd(z),
    without ever touching the channel error pattern).
    """
    H = H.to(torch.int64).clone()
    m, n = H.shape
    A = torch.cat([H, torch.eye(m, dtype=torch.int64)], dim=1)  # (m, n + m)
    pivot_cols: list[int] = []
    row = 0
    for col in range(n):
        if row >= m:
            break
        for r in range(row, m):
            if A[r, col].item() == 1:
                if r != row:
                    A[[row, r]] = A[[r, row]]
                # eliminate this column from all other rows
                for r2 in range(m):
                    if r2 != row and A[r2, col].item() == 1:
                        A[r2] = (A[r2] + A[row]) % 2
                pivot_cols.append(col)
                row += 1
                break
    if row != m:
        raise ValueError("H must have full row rank to admit a syndrome basis.")
    # E = A[:, n:] is the cumulative row-op matrix: E @ H = H_rref, with H_rref
    # carrying I_m at the pivot columns. Placing E.T at those columns of an
    # otherwise-zero (m, n) matrix gives Z with Z @ Ht = I_m.
    E = A[:, n:]
    Z = torch.zeros((m, n), dtype=torch.int64)
    Z[:, torch.tensor(pivot_cols, dtype=torch.int64)] = E.T
    # cheap sanity check on the construction (m × m mod-2 product)
    assert torch.equal(
        (Z @ H.T) % 2, torch.eye(m, dtype=torch.int64)
    ), "_syndrome_basis: Z @ H.T ≠ I_m (this is a bug)"
    return Z


class BasePerms:
    """Shared base for permutation classes. Subclasses must call
    `super().__init__(code)` and populate `self.perms` (Tensor of shape
    (n_perms, code.n)) and `self.n_perms` (int). The TTA-only `self.synd_maps`
    is built lazily on the first `sample_perms` call, so the training-time
    augmentation path never pays for the syndrome-basis solve.
    """

    code: LinearCode
    perms: Tensor
    n_perms: int
    synd_maps: Tensor | None  # lazily populated; see `compute_synd_maps`

    def __init__(self, code: LinearCode) -> None:
        self.code = code
        self.synd_maps = None

    def sample_perms(self, bs: int) -> tuple[Tensor, Tensor, Tensor]:
        """Draw `bs` random permutations and return them together with their
        inverses (both shape (bs, code.n), int64) and the corresponding
        syndrome maps (shape (bs, m, m), float32). Used by `TTADecoder` to
        permute received words, inverse-permute the resulting logits, and
        derive the input syndromes for permuted inputs from the channel
        syndrome alone — no access to the channel error pattern needed.
        """
        if self.synd_maps is None:
            self.compute_synd_maps()
        n = self.perms.size(1)
        idx = torch.randint(self.n_perms, (bs,))
        perms = self.perms[idx]
        # use scatter(arange(n)) to get the inverse permutations in one shot
        src = torch.arange(n)[None].repeat(bs, 1)
        perms_inv = torch.empty_like(perms)
        perms_inv.scatter_(dim=1, index=perms, src=src)
        synd_maps = self.synd_maps[idx]  # type: ignore[index]
        return perms, perms_inv, synd_maps

    def compute_synd_maps(self) -> None:
        """Build the linear syndrome map M_σ for every σ in `self.perms` and
        cache it in `self.synd_maps` (shape (n_perms, m, m), float32). Called
        lazily from `sample_perms` so training-only callers never trigger it.

        For any code automorphism σ, synd(σ(z)) is a linear function of
        synd(z), namely `synd(σ(z)) = synd(z) @ M_σ` mod 2 with
        `M_σ = Z[:, σ] @ Ht` where Z is a syndrome basis (Z @ Ht = I_m).
        Stored as float32 so it composes cleanly with the bipolar syndrome
        tensors used downstream.
        """
        Z = _syndrome_basis(self.code.H)  # (m, n) int64
        Ht = self.code.Ht.to(torch.int64)  # (n, m)
        perms = self.perms.to(torch.int64)  # (n_perms, n)
        # batched gather of Z's columns by each row of `perms` → (n_perms, m, n)
        Z_perm = (
            Z.unsqueeze(0)
            .expand(self.n_perms, -1, -1)
            .gather(2, perms.unsqueeze(1).expand(-1, self.code.m, -1))
        )
        self.synd_maps = ((Z_perm @ Ht) % 2).to(torch.float32)  # (n_perms, m, m)

    def assert_valid_automorphisms(self, code: LinearCode) -> None:
        """Verify that every row of `self.perms` is a code automorphism of
        `code`. A permutation σ is a code automorphism iff σ(c) is a codeword
        for every codeword c, equivalently iff σ(g) is a codeword for every
        row g of a generator matrix G — so testing on the k rows of G is both
        necessary and sufficient (cheaper and stronger than testing against
        random codewords).

        Used by `GenericPerms` as an `__init__`-time guardrail against bad
        permutation files.
        """
        G = code.G.to(torch.int64)
        Ht = code.Ht.to(torch.int64)
        perms = self.perms.to(torch.int64)
        # G[:, perms[i]] @ Ht for every i, in one shot — shape (n_perms, k, m)
        G_perm = (
            G.unsqueeze(0)
            .expand(self.n_perms, -1, -1)
            .gather(2, perms.unsqueeze(1).expand(-1, G.size(0), -1))
        )
        synds = (G_perm @ Ht) % 2
        bad = synds.reshape(self.n_perms, -1).any(dim=1)
        if bad.any():
            n_bad = int(bad.sum())
            first = int(bad.nonzero(as_tuple=True)[0][0])
            raise ValueError(
                f"{n_bad}/{self.n_perms} permutations are not code automorphisms "
                f"of {code} (first bad index: {first})."
            )


class BCHPerms(BasePerms):
    """Cyclic × Frobenius permutations for BCH codes.

    Builds the subgroup of automorphisms generated by the cyclic shift and the
    Frobenius map (`i → 2·i mod n`), of size `n · log2(n+1)`. For extended BCH
    codes (`is_extended=True`), each permutation is padded with the fixed
    overall-parity bit at the end (this only covers a small subset of the full
    automorphism group of the eBCH code; for the full group, use `GenericPerms`
    with a precomputed `.mat` file).
    """

    def __init__(self, code: LinearCode, is_extended: bool = False) -> None:
        super().__init__(code)
        # precompute the subgroup of cyclic x frobenius permutations
        n = code.n - 1 if is_extended else code.n
        idx = torch.arange(
            n, dtype=torch.int64
        )  # take_along_dim requires int64 indices
        b = torch.log2(torch.tensor(n + 1, dtype=torch.int)).int()
        rows = [(j + idx * 2**l) % n for j in range(n) for l in range(b)]
        perms = torch.stack(rows)
        self.n_perms = perms.size(0)
        if is_extended:
            # add the fixed extension bit to the end of each permutation
            perms = torch.hstack(
                [perms, n * torch.ones(self.n_perms, 1, dtype=torch.int64)]
            )
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


class QCPerms(BasePerms):
    """Quasi-cyclic shift permutations for QC-LDPC codes.

    Builds the `Zc` permutations corresponding to a cyclic shift by 0, 1, ...,
    Zc-1 within each circulant block of size `Zc`. Requires that `Zc` divides
    `code.n`.
    """

    def __init__(self, code: LinearCode, Zc: int = 1) -> None:
        super().__init__(code)
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


class GenericPerms(BasePerms):
    """Load an arbitrary set of code automorphisms from a `.mat` file.

    Useful for codes whose automorphism group is not directly captured by
    `BCHPerms` or `QCPerms` — e.g. Reed-Muller codes (affine group) or polar
    codes (factor-graph automorphisms).

    Expected `.mat` file structure
    ------------------------------
    The file must contain a single field `perms` of shape `(n_perms, code.n)`.
    Each row must be a valid permutation of `[0, 1, ..., code.n - 1]`. Both
    MATLAB v7 and v7.3 (HDF5) formats are supported.

    Two example files are shipped under `data/perms/`:
      - `perms.rm.32.mat`    — 1024 affine-group permutations of length 32,
                               valid for any RM(32, k) code.
      - `perms.polar.128.mat` — 4096 permutations of length 128 from UTA(8),
                                valid for any polar(128, k) code constructed
                                from Arikan's [1 0; 1 1] kernel.

    Arguments
    ---------
    code      : LinearCode whose code length must match the second dim of `perms`.
    mat_file  : path to the .mat file. The `.mat` extension is appended if missing.
    num_perms : optional. If set and smaller than the number of rows in the file,
                only the first `num_perms` rows are kept (lets you study how TTA /
                training-augmentation performance scales with the size of the
                automorphism subset). If unset (default), all rows are loaded.
    """

    def __init__(
        self, code: LinearCode, mat_file: str, num_perms: int | None = None
    ) -> None:
        super().__init__(code)
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
        # Externally supplied perms are not correct by construction (unlike
        # BCHPerms / QCPerms): verify each one is a genuine code automorphism
        # before accepting the file. Synd-map construction stays lazy.
        self.assert_valid_automorphisms(code)
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
