# Lightning data module for SBND models training and testing

import math, torch, numpy as np
import h5py  # type: ignore[import-untyped]
from typing import Any, Iterable, Callable, cast
from omegaconf import OmegaConf, ListConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from lightning import LightningDataModule
from scipy.io import loadmat  # type: ignore[import-untyped]
from .codes import LinearCode
from .utils import get_rank_zero_logger

log = get_rank_zero_logger(__name__)


def build_batch_mix(K: int, weights: list[float] | None, name: str = "items") -> Tensor:
    """Normalize `batch_mix` weights into a (K,) tensor that sums to 1.
    Uniform weights are used when `weights is None`. Validates length match,
    non-negativity, and positive sum. `name` is the user-facing label of the
    group axis (e.g. "ebno_dB" or "train_file"), used only in error messages."""
    if K == 0:
        raise ValueError(f"{name} must contain at least one value")
    if weights is None:
        return torch.full((K,), 1.0 / K, dtype=torch.float32)
    if len(weights) != K:
        raise ValueError(
            f"batch_mix length ({len(weights)}) must match {name} length ({K})"
        )
    w_t = torch.tensor([float(x) for x in weights], dtype=torch.float32)
    if torch.any(w_t < 0):
        raise ValueError(f"batch_mix must be non-negative; got {weights}")
    w_sum = w_t.sum()
    if w_sum.item() <= 0:
        raise ValueError(f"batch_mix must have a positive sum; got {weights}")
    return w_t / w_sum


def build_loss_weights(weights: list[float] | None, K: int) -> Tensor:
    """Normalize `loss_weights` into a (K,) tensor of non-negative floats with
    at least one positive entry. Defaults to all-ones (no reweighting).
    Unlike `batch_mix`, this is NOT normalized to sum to 1 — relative scale is
    eliminated downstream by the weighted-mean loss formula in the
    SBNDLitModule."""
    if weights is None:
        # default to all-ones (no reweighting)
        return torch.ones(K, dtype=torch.float32)
    if len(weights) != K:
        raise ValueError(
            f"loss_weights length ({len(weights)}) must match group count ({K})"
        )
    w = torch.tensor([float(x) for x in weights], dtype=torch.float32)
    if torch.any(w < 0):
        raise ValueError(f"loss_weights must be non-negative; got {weights}")
    if w.sum().item() <= 0:
        raise ValueError(f"loss_weights must have a positive sum; got {weights}")
    return w


def to_float_list(x: Any) -> list[float] | None:
    """Coerce a scalar / list / tuple / OmegaConf ListConfig / None into a
    `list[float]` (or None). Used to normalize Hydra/YAML-typed args before
    handing them off to plain-Python downstream classes. Items that cannot be
    cast to float raise a `ValueError` here, at the DataModule boundary, rather
    than deeper in the dataset where the original argument name is lost."""
    if x is None:
        return None
    if isinstance(x, ListConfig):
        x = OmegaConf.to_object(x)
    if isinstance(x, (list, tuple)):
        try:
            return [float(v) for v in x]
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"could not coerce {list(x)!r} to a list of floats — "
                "check that list items are comma-separated in YAML "
                "(e.g. `[3.0, 4.0]`, not `[3.0 4.0]`)"
            ) from e
    return [float(x)]


def to_string_list(x: Any) -> list[str] | None:
    """Coerce a scalar / list / tuple / OmegaConf ListConfig / None into a
    `list[str]` (or None). Same role as `to_float_list` but for path-shaped
    args like `train_file`. Hydra interpolations such as `${data_dir}/foo.mat`
    inside list items are resolved by OmegaConf before we see them here."""
    if x is None:
        return None
    if isinstance(x, ListConfig):
        x = OmegaConf.to_object(x)
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    return [str(x)]


def load_matlab_data(mat_file: str) -> tuple[Tensor, Tensor]:
    """Load received words y and target error patterns e from a MATLAB .mat file."""
    try:
        # v7 mat files or earlier are supported by scipy.io
        matlab_data = loadmat(mat_file, squeeze_me=True)
    except NotImplementedError:
        # but not v7.3 (=HDF5) mat files, for which we need h5py
        with h5py.File(mat_file, "r") as f:
            if "y" not in f or "e" not in f:
                raise ValueError(f"Datasets 'y' and 'e' not found in {mat_file}")
            y = torch.from_numpy(f["y"][:].astype(np.float32).transpose())
            e = torch.from_numpy(f["e"][:].astype(np.int8).transpose())
    else:
        if "y" not in matlab_data or "e" not in matlab_data:
            raise ValueError(f"Datasets 'y' and 'e' not found in {mat_file}")
        y = torch.tensor(matlab_data["y"], dtype=torch.float32)
        e = torch.tensor(matlab_data["e"], dtype=torch.int8)
    assert e.dtype == torch.int8 and y.dtype == torch.float32
    return y, e


def prepare_data(
    code: LinearCode, y: Tensor, e: Tensor, error_space: str = "codeword"
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Prepare a batch of training samples for the SBND decoder: create the input pair
    x =(|y|,s) from a received word y and associated binary error pattern e.
    Return the triplet (|y|, s, e), with s in bipolar format (0 -> +1, 1 -> -1).
    If `error_space` is "message", the returned error pattern is mapped to message
    space as e_msg = (e @ Ginv) mod 2 (shape (bs, k)); otherwise it is left as-is
    in codeword space (shape (bs, n)).
    Output shape: (bs, n), (bs, m), (bs, n or k) for an (n,k) code with m parity-check eqs
    """
    ym = torch.abs(y)
    ym = ym / torch.max(ym, dim=1)[0].unsqueeze(
        1
    )  # normalize magnitude to [0,1] within each rx word
    s = code.syndrome(e)
    if error_space == "message":
        e = (e @ code.Ginv).bitwise_and(1)
    return ym, (1 - 2 * s).float(), e


def proportional_counts(
    weights: Tensor, bs: int, group_labels: list[str] | Tensor | None = None
) -> Tensor:
    """Deterministic per-group sample counts.

    Returns a 1D long tensor `counts` of same length as `weights` such that
    `counts.sum() == bs` and the relative proportions match `weights` as
    closely as possible (proportional rounding: floor everything, then
    distribute the remainder by largest fractional part).

    Any group with `weights[k] > 0` whose count would round to zero is bumped
    to 1 — so every requested group is guaranteed at least one sample per
    batch — and a one-shot warning is emitted naming the affected groups (if
    `group_labels` is provided). The bumped samples are compensated by
    decrementing the largest counts among the other groups.
    """
    assert weights.dim() == 1 and bs > 0
    raw = weights * bs
    counts = raw.floor().long()

    # standard proportional rounding to make sum == bs
    remainder = bs - int(counts.sum().item())
    if remainder > 0:
        _, idx = torch.topk(raw - counts.float(), remainder)
        counts[idx] += 1

    # bump tiny non-zero weights to 1 sample/batch, decrement elsewhere to compensate
    tiny_mask = (weights > 0) & (counts == 0)
    n_tiny = int(tiny_mask.sum().item())
    if n_tiny > 0:
        counts[tiny_mask] = 1
        # decrement the largest non-tiny counts; exclude tiny entries from the topk
        candidates = counts.clone()
        candidates[tiny_mask] = -1
        if int(candidates.max().item()) < 2:
            raise ValueError(
                f"cannot allocate at least 1 sample per batch to all {weights.numel()} "
                f"requested groups at bs={bs}: weights {weights.tolist()} are too dispersed. "
                "Increase batch size, drop some groups, or merge their weights."
            )
        _, idx = torch.topk(candidates, n_tiny)
        counts[idx] -= 1
        if group_labels is not None:
            if isinstance(group_labels, Tensor):
                tiny_labels = group_labels[tiny_mask].tolist()
            else:
                tiny_labels = [
                    group_labels[i]
                    for i in range(len(group_labels))
                    if bool(tiny_mask[i])
                ]
            log.warning(
                f"Group(s) {tiny_labels} have weights too small to yield a sample "
                f"per batch at bs={bs}; bumped to 1 sample/batch each."
            )
    return counts


def generate_at_single_snr(
    code: LinearCode, sigma: float, n: int
) -> tuple[Tensor, Tensor]:
    """Rejection sampling at one SNR (`sigma` is the AWGN std at that SNR)
    until exactly `n` non-zero-syndrome rows have been accumulated.
    Output shape: (n, code.n), (n, code.n)."""
    n_samples = 0
    y = torch.empty(0, code.n, dtype=torch.float32)
    e = torch.empty(0, code.n, dtype=torch.int8)
    while n_samples < n:
        y_ = 1 + sigma * torch.randn(n, code.n)
        e_ = (y_ < 0).to(torch.int8)
        s_ = code.syndrome(e_)
        nz_synd_idx = torch.any(s_, dim=1).nonzero().squeeze(1)
        n_to_add = min(n - n_samples, nz_synd_idx.shape[0])
        if n_to_add > 0:
            nz_rows = nz_synd_idx[:n_to_add]
            y = torch.cat((y, y_[nz_rows]))
            e = torch.cat((e, e_[nz_rows]))
            n_samples += n_to_add
    return y, e


def generate_random_training_batch(
    code: LinearCode,
    ebno_dB: Tensor,
    counts: Tensor,
    w_per_group: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Generate a batch of training samples with **constant** per-SNR counts.

    `ebno_dB` is a 1D tensor of SNR points and `counts` is a 1D tensor of same
    length giving the number of non-zero-syndrome samples to draw at each SNR.
    The output batch size is `counts.sum()`. The per-SNR rejection sampling is
    decoupled so the non-zero syndrome filter does not skew the empirical
    distribution of kept samples — every batch hits the requested proportions
    exactly. Use `proportional_counts(weights, bs)` to derive `counts` from a
    target weight distribution; cache the result if `bs` and `weights` are
    fixed (which they are inside the on-demand datasets).

    `w_per_group` optionally gives a per-SNR scalar loss weight; the returned
    per-sample weight tensor `w` repeats `w_per_group[k]` `counts[k]` times,
    then gets the same row permutation as `y` and `e`. Defaults to all-ones.

    Per-SNR blocks are shuffled into a random row order before returning, so
    callers never see contiguous same-SNR rows. Transmission of the all-zero
    codeword is assumed; all returned rows have non-zero syndrome.

    Output shape: (bs, n), (bs, n), (bs,).
    """
    assert (
        ebno_dB.dim() == 1 and counts.dim() == 1 and ebno_dB.numel() == counts.numel()
    )
    y_parts: list[Tensor] = []
    e_parts: list[Tensor] = []
    w_parts: list[Tensor] = []
    if w_per_group is None:
        w_per_group = torch.ones(ebno_dB.numel(), dtype=torch.float32)
    for k in range(ebno_dB.numel()):
        n_k = int(counts[k].item())
        if n_k == 0:
            continue
        sigma_k = 1 / math.sqrt(2 * code.rate * 10 ** (float(ebno_dB[k].item()) / 10))
        y_k, e_k = generate_at_single_snr(code, sigma_k, n_k)
        y_parts.append(y_k)
        e_parts.append(e_k)
        w_parts.append(w_per_group[k].expand(n_k))
    y = torch.cat(y_parts)
    e = torch.cat(e_parts)
    w = torch.cat(w_parts)
    perm = torch.randperm(y.shape[0])
    return y[perm], e[perm], w[perm]


def generate_random_test_batch(
    code: LinearCode, ebno_dB: float, bs: int
) -> tuple[Tensor, Tensor]:
    """
    Generate a random batch of bs test samples for the given code and Eb/N0 value (dB)
    Each sample is a tuple (y, e) where:
        - y is the received vector
        - e is the target binary error pattern introduced by the channel
    The rx words are noisy observations of randomly generated codewords.
    No filtering is applied to the generated samples, which can include rx words with a zero syndrome.
    Output shape is (bs, n), (bs, n)
    """
    sigma = 1 / math.sqrt(2 * code.rate * 10 ** (ebno_dB / 10))
    cw = code.encode(torch.randint(2, (bs, code.k), dtype=torch.int8))
    y = 1 - 2 * cw + sigma * torch.randn(bs, code.n)
    e = (y < 0).to(torch.int8).bitwise_xor_(cw)  # e = z - c
    return y, e


class OnDemandDataset(Dataset):
    """
    Simulate a fixed dataset where each of the n_batches items is a batch of size bs.
    Batches are generated randomly, one at a time. The data never repeat between epochs.
    To be used with a DataLoader with automatic batching disabled (batch_size=None)
    If `train=True`, the batches are made of rx words with non-zero syndromes only.

    `ebno_dB` may be a single float or a list of floats giving the Eb/N0
    distribution to sample from. The optional `batch_mix` argument controls the
    proportion of samples drawn at each SNR (defaults to uniform; values are
    normalized to sum to 1). The optional `loss_weights` argument attaches a
    per-SNR scalar loss weight to each sample (defaults to all-ones).

    Test mode requires a single `ebno_dB` value.

    Each `__getitem__` returns a 4-tuple `(ym, s, e, w)` where `w` is a `(bs,)`
    per-sample loss weight (constant per-group, replicated by `counts`).
    """

    ebno_dB: Tensor
    batch_mix: Tensor
    loss_weights: Tensor
    counts: Tensor | None

    def __init__(
        self,
        code: LinearCode,
        ebno_dB: float | list[float],
        n_batches: int,
        bs: int,
        train: bool = False,
        error_space: str = "codeword",
        batch_mix: list[float] | None = None,
        loss_weights: list[float] | None = None,
    ) -> None:
        self.code = code
        self.n_batches, self.bs = n_batches, bs
        self.train = train
        self.error_space = error_space

        ebno_list = (
            [float(ebno_dB)]
            if not isinstance(ebno_dB, (list, tuple))
            else [float(v) for v in ebno_dB]
        )
        self.batch_mix = build_batch_mix(len(ebno_list), batch_mix, name="ebno_dB")
        self.ebno_dB = torch.tensor(ebno_list, dtype=torch.float32)
        self.loss_weights = build_loss_weights(loss_weights, len(ebno_list))
        if not train and self.ebno_dB.numel() != 1:
            raise ValueError(
                f"OnDemandDataset in test mode requires a single ebno_dB; got {self.ebno_dB.tolist()}"
            )
        # cache per-SNR sample counts within a batch (training only; test mode
        # goes through generate_random_test_batch which doesn't use counts)
        self.counts = (
            proportional_counts(self.batch_mix, self.bs, self.ebno_dB.tolist())
            if train
            else None
        )

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.train:
            assert self.counts is not None  # set by __init__ when train=True
            y, e, w = generate_random_training_batch(
                self.code, self.ebno_dB, self.counts, self.loss_weights
            )
            ym, s, e = prepare_data(self.code, y, e, self.error_space)
            return ym, s, e, w
        else:
            # test mode is constrained to a single SNR by __init__
            y, e = generate_random_test_batch(
                self.code, float(self.ebno_dB[0].item()), self.bs
            )
            ym, s, e = prepare_data(self.code, y, e, self.error_space)
            w = torch.ones(self.bs, dtype=torch.float32)
            return ym, s, e, w

    def __len__(self) -> int:
        return self.n_batches


class SBNDDataset(Dataset):
    """
    Create a dataset for SBND decoding of the linear code `code`,
    from the pair of tensors (y, e).

    Each sample is returned as a 4-tuple `(ym, s, e, w)` where `w` is the
    per-sample loss weight (a scalar tensor of value `w_per_sample`, constant
    across rows of this dataset). Use `w_per_sample` to inject the per-dataset
    `loss_weights` value when mixing several files.

    This is essentially a wrapper around Pytorch TensorDataset:
    - https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataset.py#L189

    If `transform` is not None, the specified transform is applied to the data at dataloading.
    """

    def __init__(
        self,
        code: LinearCode,
        y: Tensor,
        e: Tensor,
        transform: Callable | None = None,
        error_space: str = "codeword",
        w_per_sample: float = 1.0,
    ) -> None:
        assert y.shape[0] == e.shape[0], "y and e must have same number of samples"
        self.code = code
        self.transform = transform
        self.error_space = error_space
        self.y, self.e = y, e
        self.w_per_sample = float(w_per_sample)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # `prepare_data` expects a (bs, n) batch; wrap the single row in a
        # 1-row batch and squeeze the result back. (DataLoader normally hits
        # `__getitems__` instead, but single-sample access matters when this
        # dataset is wrapped in a higher-level structure.)
        y_row = self.y[index].unsqueeze(0)
        e_row = self.e[index].unsqueeze(0)
        if self.transform is not None:
            y_row, e_row = self.transform(y_row, e_row)
        ym, s, e = prepare_data(self.code, y_row, e_row, self.error_space)
        w = torch.tensor(self.w_per_sample, dtype=torch.float32)
        return ym.squeeze(0), s.squeeze(0), e.squeeze(0), w

    def __getitems__(
        self, indices: list[int]
    ) -> list[tuple[Tensor, Tensor, Tensor, Tensor]]:
        if self.transform is not None:
            y_t, e_t = self.transform(self.y[indices], self.e[indices])
            ym, s, e = prepare_data(self.code, y_t, e_t, self.error_space)
        else:
            ym, s, e = prepare_data(
                self.code, self.y[indices], self.e[indices], self.error_space
            )
        n = ym.size(0)
        w_scalar = torch.tensor(self.w_per_sample, dtype=torch.float32)
        return [(ym[i], s[i], e[i], w_scalar) for i in range(n)]

    def __len__(self) -> int:
        return self.y.size(0)


class MultiDatasetTrainDataset(Dataset):
    """
    Produces batches mixing rows from K `SBNDDataset` files in fixed per-batch
    proportions. Each `__getitem__(b)` returns a full batch `(ym, s, e, w)` of
    size `bs = counts.sum()`, with `counts[k]` rows drawn from dataset k and
    rows then permuted into a random order so consumers never see contiguous
    same-source blocks.

    Per-sample loss weights are inherited from each dataset's `w_per_sample`
    (so callers must set `loss_weights[k]` on `datasets[k]` at construction).

    Epoch lifecycle and DDP correctness:

        Per-dataset shuffled index lists are derived deterministically from
        `(base_seed, epoch, k)` at every epoch via `set_epoch(epoch)`. The
        datamodule calls `set_epoch` from `on_train_epoch_start`, so all DDP
        ranks compute identical shuffles and Lightning's auto-installed
        `DistributedSampler` over batch indices `[0, n_batches)` gives each
        rank a disjoint slice — no row appears in two batches of the same
        epoch, across all ranks. No custom Sampler is required.

        The intra-batch row permutation uses the global RNG (worker-decorrelated
        by `seed_everything(workers=True)`); this affects only display order
        within a batch, not sample identity, so per-worker randomness is fine.

    Caveat: `persistent_workers=True` is not supported — when workers persist
    across epochs they don't see `set_epoch` updates made in the main process.
    The DataModule logs a warning if it detects this setting.

    Epoch length is `min_k floor(len(datasets[k]) / counts[k])`. Larger files
    are partially traversed each epoch (their unused rows are seen on later
    epochs after reshuffle); a per-dataset utilization summary is logged at
    startup so this is visible.
    """

    def __init__(
        self,
        datasets: list[SBNDDataset],
        counts: Tensor,
        base_seed: int = 0,
    ) -> None:
        # Note: `datasets` may also contain `Subset[SBNDDataset]` when a
        # validation split was taken from the first file. Subset exposes
        # __getitems__ (delegating through self.indices), so the batched
        # fetch in `__getitem__` works uniformly. The list[SBNDDataset]
        # annotation is the dominant case; the Subset substitution is
        # tolerated at the one call site that does it.
        assert len(datasets) == counts.numel() and len(datasets) >= 1
        self.datasets = datasets
        self.counts = counts.long()
        self.base_seed = int(base_seed)
        # n_batches/epoch limited by the dataset that runs out first under
        # its share of the batch; see class docstring for rationale
        self.n_batches = min(
            len(d) // int(c) for d, c in zip(datasets, self.counts.tolist())
        )
        self._epoch = -1
        self._perms: list[Tensor] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle per-dataset index lists deterministically for `epoch`.
        Must be called from the main process before iteration starts (so workers
        inherit the new state at spawn-time via pickle)."""
        if epoch == self._epoch:
            return
        self._epoch = int(epoch)
        self._perms = []
        for k, d in enumerate(self.datasets):
            g = torch.Generator()
            # spread bits across (base_seed, epoch, k) to avoid trivial collisions
            g.manual_seed(self.base_seed + 1_000_003 * (self._epoch + 1) + 31 * k)
            self._perms.append(torch.randperm(len(d), generator=g))

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, b: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ym_parts: list[Tensor] = []
        s_parts: list[Tensor] = []
        e_parts: list[Tensor] = []
        w_parts: list[Tensor] = []
        for k, d in enumerate(self.datasets):
            c = int(self.counts[k].item())
            idx = self._perms[k][b * c : (b + 1) * c].tolist()
            items = d.__getitems__(idx)
            ym_parts.append(torch.stack([it[0] for it in items]))
            s_parts.append(torch.stack([it[1] for it in items]))
            e_parts.append(torch.stack([it[2] for it in items]))
            w_parts.append(torch.stack([it[3] for it in items]))
        ym = torch.cat(ym_parts)
        s = torch.cat(s_parts)
        e = torch.cat(e_parts)
        w = torch.cat(w_parts)
        # intra-batch row shuffle (worker-local RNG; safe — affects only order)
        perm = torch.randperm(ym.size(0))
        return ym[perm], s[perm], e[perm], w[perm]


class SBNDDataModule(LightningDataModule):
    """
    DataModule for SBND training and testing.

    Two training modes are supported, selected by whether at least one training
    dataset file is specified (`train_file` set) or not (the default):

    1. On-demand (`train_file` unset):
        - training and validation use freshly-generated on-demand data
        - data generation is performed at the Eb/N0 value(s) given by `ebno_dB_train` (must be set)
        - `n_train_samples` will be used for training (must be specified explicitly)
        - `n_val_samples` will be used for validation (default value = 25% of `n_train_samples`)
        - `n_train_samples` (resp. `n_val_samples`) is rounded down to a multiple of `train_bs`
          (resp. `val_bs`) if necessary, as on-demand data generation requires data size to be
          a multiple of batch size
        - no data augmentation is applied, even if `transform` is not None

    2. Pre-computed dataset(s) (`train_file` set):
        - training uses the first `n_train_samples` of each listed file (entire file if 0)
        - `train_file` may be a single path or a list of paths; with a list, each batch is a
          fixed mixture of rows from the files (proportions controlled by `batch_mix`,
          defaults to uniform). Epoch length is `min_k floor(N_k / counts[k])`; larger files
          are partially traversed each epoch
        - if `transform` is not None, the specified augmentation is applied to all loaded
          training rows (validation samples are never augmented)
        - if `val_file` is given, validation uses the first `n_val_samples` of that file
          (entire file if 0); otherwise a validation set of `n_val_samples` is created by
          random split of the FIRST training file (default split = 75/25 unless
          `n_val_samples` is explicit)

    Both modes share two optional knobs over the K-element group axis (K = number of SNR
    points in mode 1, number of train files in mode 2):

        - `batch_mix: list[float] | None` — per-batch proportions across groups; normalized
          to sum to 1; defaults to uniform. In on-demand mode, this controls the SNR distribution
          within each training batch; in dataset mode, this controls the proportion of examples
          drawn from each file into each batch.
        - `loss_weights: list[float] | None` — per-sample loss multiplier per group; default
          all-ones. Combined with the per-sample weighted-mean loss in `SBNDLitModule` to
          implement Wiesmayr-style per-group reweighting (see docs/training.md for the
          formula and its coupling with `batch_mix`).

    On-demand validation must run at a single SNR to be meaningful: `ebno_dB_val` selects
    it. When `ebno_dB_train` is a single float, `ebno_dB_val` defaults to that value
    (backward compatible); when `ebno_dB_train` is a list, `ebno_dB_val` must be specified
    explicitly. In dataset mode, validation comes from the file (split or `val_file`) and
    `ebno_dB_val` has no effect.

    Test data is always generated on the fly, and can include rx words with a zero syndrome.
    It is possible to pass a list of Eb/N0 values to `ebno_dB_test`, in which case different
    test datasets will be generated, one per SNR value, of size `n_test_samples` each.
    Here also `n_test_samples` is rounded down to the nearest multiple of `test_bs` if needed.

    Extra arguments can be passed to pytorch DataLoader, e.g. `num_workers`, through
    `extra_args`. Note: with multi-file dataset training (K >= 2), `persistent_workers=True`
    is not supported — see `MultiDatasetTrainDataset` for details.

    Reproducibility and sample-correlation caveats (on-demand mode):
        On-demand training, validation and test batches are produced by the global
        `torch` RNG inside `generate_random_*_batch`. Sample identity therefore
        depends entirely on the RNG state of each DataLoader worker process and of
        each DDP rank. Two independent failure modes exist:
            1. Workers within a rank drawing from the same RNG stream → duplicated
               samples across workers (single-GPU, multi-worker).
            2. DDP ranks drawing from the same RNG stream → duplicated samples
               across GPUs, which silently negates the benefit of multi-GPU
               training without raising any error.
        Both are avoided by calling
            `lightning.pytorch.seed_everything(seed, workers=True)`
        once before `Trainer.fit` / `Trainer.test`. The `workers=True` flag installs
        a `worker_init_fn` that reseeds each worker with `base_seed + worker_id`,
        and Lightning additionally offsets `base_seed` per DDP rank. If no seed is
        configured at all, decorrelation is still preserved in practice because
        each DDP rank is launched as a fresh subprocess (independent OS-entropy
        seed) and PyTorch's default DataLoader assigns each worker a distinct
        derived seed; only run-to-run reproducibility is lost.
        The dangerous misuse to avoid is calling `seed_everything(seed)` *without*
        `workers=True`: ranks are then offset but workers within a rank share an
        identical RNG stream, producing correlated batches.
    """

    train_ds: Dataset
    val_ds: Dataset
    test_ds: list[Dataset]
    ebno_dB_train: list[float] | None
    batch_mix: list[float] | None
    loss_weights: list[float] | None
    ebno_dB_val: float | None
    ebno_dB_test: list[float]
    train_files: list[str] | None

    def __init__(
        self,
        code: LinearCode,
        ebno_dB_train: float | Iterable[float] | None = None,
        train_file: str | Iterable[str] | None = None,
        n_train_samples: int = 0,
        train_bs: int = 1024,
        val_file: str | None = None,
        n_val_samples: int = 0,
        val_bs: int = 1024,
        ebno_dB_val: float | None = None,
        ebno_dB_test: float | Iterable[float] = 0.0,
        n_test_samples: int = 2**20,
        test_bs: int = 1024,
        batch_mix: Iterable[float] | None = None,
        loss_weights: Iterable[float] | None = None,
        transform: Callable | None = None,
        error_space: str = "codeword",
        extra_args: dict | None = None,
        base_seed: int = 0,
    ) -> None:
        super().__init__()

        self.code = code
        log.info(f"Instantiating an SBNDDataModule for the {code} code")

        # mode dispatch:
        #   - train_file unset → on-demand
        #   - train_file set   → pre-computed dataset(s)
        self.train_files = to_string_list(train_file)
        self.on_demand = self.train_files is None

        if self.on_demand:
            log.info("Training uses on-demand data")
            if ebno_dB_train is None:
                raise ValueError(
                    "At least one training Eb/N0 value ebno_dB_train must be specified"
                )
            if n_train_samples == 0:
                raise ValueError(
                    "The number of training samples n_train_samples must be specified and > 0"
                )
            if val_file is not None:
                raise ValueError(
                    "val_file cannot be specified without train_file "
                    "(on-demand mode does not support validation set files)"
                )
        else:
            assert self.train_files is not None
            if len(self.train_files) == 1:
                log.info(f"Training uses the fixed dataset {self.train_files[0]}")
            else:
                log.info(
                    f"Training uses {len(self.train_files)} fixed datasets, mixed within each batch: "
                    + ", ".join(self.train_files)
                )
            if ebno_dB_train is not None:
                log.warning(
                    "ebno_dB_train is set but training uses fixed dataset file(s); "
                    "the value will be ignored"
                )
            if ebno_dB_val is not None:
                log.warning(
                    "ebno_dB_val is set but training uses fixed dataset file(s); "
                    "the value will be ignored"
                )

        # store non-SNR arguments
        self.n_train_samples, self.train_bs = n_train_samples, train_bs
        self.val_file, self.n_val_samples, self.val_bs = val_file, n_val_samples, val_bs
        self.n_test_samples, self.test_bs = n_test_samples, test_bs
        self.transform = transform(code) if transform is not None else None
        self.error_space = error_space
        self.extra_args = extra_args if extra_args is not None else {}
        self.base_seed = int(base_seed)
        self.save_hyperparameters(
            logger=False
        )  # snapshots __init__ args for ckpt; safe to mutate self attrs below

        # warn loudly if persistent_workers=True in extra_args and we'll be in
        # the multi-file mode (set_epoch updates wouldn't propagate to workers)
        if (
            not self.on_demand
            and self.train_files is not None
            and len(self.train_files) >= 2
            and self.extra_args.get("persistent_workers", False)
        ):
            log.warning(
                "persistent_workers=True is not supported with multi-file training "
                "(per-epoch dataset shuffles won't propagate to workers). "
                "Force-disabling it for the training DataLoader."
            )
            # we'll honor this in train_dataloader() by stripping the flag

        # coerce list-shaped Hydra args (ListConfig/list/tuple/scalar) into
        # plain `list[float]`. Validation of length and non-negativity happens
        # inside the dataset classes via `build_group_dist` / `build_loss_weights`.
        self.ebno_dB_test = to_float_list(ebno_dB_test) or [0.0]
        self.ebno_dB_train = to_float_list(ebno_dB_train)
        self.batch_mix = to_float_list(batch_mix)
        self.loss_weights = to_float_list(loss_weights)

        # resolve the validation SNR — single value required to be meaningful.
        # Only used in on-demand mode (dataset mode derives validation
        # from the first training file). Defaults to the training SNR when that
        # is a single value (backward compatible); must be specified explicitly
        # when training mixes several SNRs.
        if self.on_demand:
            assert self.ebno_dB_train is not None
            if ebno_dB_val is not None:
                self.ebno_dB_val = float(ebno_dB_val)
            elif len(self.ebno_dB_train) == 1:
                self.ebno_dB_val = self.ebno_dB_train[0]
            else:
                raise ValueError(
                    "ebno_dB_val must be specified when ebno_dB_train is a list "
                    "of multiple values (validation requires a single SNR)"
                )
        else:
            self.ebno_dB_val = None

    def _load_ds(
        self,
        mat_file: str,
        n_samples: int,
        train: bool = False,
        w_per_sample: float = 1.0,
    ) -> tuple[SBNDDataset, int]:
        y, e = load_matlab_data(mat_file)
        assert (
            y.shape == e.shape and y.shape[1] == self.code.n
        ), f"y and e must have shape (n_samples, {self.code.n}); got y={tuple(y.shape)}, e={tuple(e.shape)}"
        n_samples_in_file = y.shape[0]
        if n_samples == 0:
            n_samples = n_samples_in_file
        else:
            n_samples = min(n_samples, n_samples_in_file)
            y, e = y[:n_samples], e[:n_samples]  # shrink the data if needed
        transform = (
            self.transform if train else None
        )  # apply transforms to training data only
        return (
            SBNDDataset(
                self.code, y, e, transform, self.error_space, w_per_sample=w_per_sample
            ),
            n_samples,
        )

    def _setup_dataset_training(self) -> None:
        """Set up `self.train_ds` and `self.val_ds` from `self.train_files`.
        Handles both the single-file (K=1, backcompat path) and multi-file
        (K>=2) cases, plus the validation split / `val_file` logic."""
        assert self.train_files is not None
        K = len(self.train_files)

        # per-group loss weights (length K, default all-ones)
        loss_w = build_loss_weights(self.loss_weights, K)
        # per-group batch proportions (length K, default uniform), only used for K>=2
        batch_mix_t = build_batch_mix(K, self.batch_mix, name="train_file")

        # load each training file as its own SBNDDataset, stamped with its
        # per-dataset loss weight; the validation split below shares (y, e)
        # storage with the FIRST loaded dataset
        loaded: list[SBNDDataset] = []
        per_file_n: list[int] = []
        for k, path in enumerate(self.train_files):
            ds, n = self._load_ds(
                path,
                self.n_train_samples,
                train=True,
                w_per_sample=float(loss_w[k].item()),
            )
            loaded.append(ds)
            per_file_n.append(n)
            log.info(f"Loaded {n} samples from training file: {path}")

        # ----- validation set (always built from the FIRST training file) -----
        first_ds = loaded[0]
        first_n = per_file_n[0]

        if self.val_file is not None:
            self.val_ds, self.n_val_samples = self._load_ds(
                self.val_file, self.n_val_samples, train=False
            )
            log.info(
                f"Loaded {self.n_val_samples} samples from the validation set file: {self.val_file}"
            )
        else:
            if self.n_val_samples == 0:
                self.n_val_samples = first_n // 4  # default split ratio = 75/25
            if self.n_val_samples >= first_n:
                raise ValueError(
                    f"n_val_samples ({self.n_val_samples}) must be < the number of loaded "
                    f"training samples in the first file ({first_n})"
                )
            train_n = first_n - self.n_val_samples
            train_subset, val_subset = random_split(
                first_ds, [train_n, self.n_val_samples]
            )
            # validation gets a sibling dataset with the same (y, e) storage but
            # no transform and unit loss weight (validation is never reweighted)
            val_base = SBNDDataset(
                self.code,
                first_ds.y,
                first_ds.e,
                transform=None,
                error_space=self.error_space,
                w_per_sample=1.0,
            )
            self.val_ds = Subset(val_base, list(val_subset.indices))
            # replace first training dataset with the train-only Subset (so multi-file
            # mode draws its first-file samples from outside the validation rows).
            # Subset behaves equivalently to SBNDDataset for our access pattern
            # (forwards __getitems__), so MultiDatasetTrainDataset accepts it
            # transparently — see its docstring.
            loaded[0] = train_subset  # type: ignore[call-overload]
            per_file_n[0] = train_n
            split_ratio = train_n / (train_n + self.n_val_samples)
            log.info(
                f"Created a {100*split_ratio:.0f}%/{100*(1-split_ratio):.0f}% random split"
                f" from {self.train_files[0]!r} with {train_n} samples for training"
                f" and {self.n_val_samples} samples for validation"
            )

        # ----- training set -----
        if K == 1:
            # single-file path preserves the standard shuffled DataLoader
            # behavior (Lightning auto-wraps with DistributedSampler under DDP)
            self.train_ds = loaded[0]
            self.n_train_samples = per_file_n[0]
            if self.batch_mix is not None:
                log.warning(
                    "batch_mix is set but only one train_file is provided; ignored"
                )
            if self.loss_weights is not None:
                log.info(
                    f"Per-sample loss weight = {loss_w[0].item():.3f} for all training samples"
                )
        else:
            # multi-file path: batch-producing dataset; per-epoch shuffles
            # seeded by (base_seed, epoch) via set_epoch (called from
            # on_train_epoch_start). DDP correctness is documented in
            # MultiDatasetTrainDataset.
            counts = proportional_counts(
                batch_mix_t, self.train_bs, group_labels=self.train_files
            )
            mdt = MultiDatasetTrainDataset(loaded, counts, base_seed=self.base_seed)
            self.train_ds = mdt
            self.n_train_samples = mdt.n_batches * self.train_bs
            # per-file utilization report
            util = [
                100.0 * mdt.n_batches * int(c) / n
                for c, n in zip(counts.tolist(), per_file_n)
            ]
            mix_str = ", ".join(
                f"{path!r}: count={int(c)}/batch, "
                f"loss_w={float(loss_w[k].item()):.3g}, "
                f"epoch util={util[k]:.0f}%"
                for k, (path, c) in enumerate(zip(self.train_files, counts.tolist()))
            )
            log.info(
                f"Created a multi-file training mix over {K} datasets: "
                f"{mdt.n_batches} batches/epoch ({self.n_train_samples} samples/epoch); "
                f"per-file: {mix_str}"
            )

    def _setup_on_demand_training(self) -> None:
        assert self.ebno_dB_train is not None
        assert self.ebno_dB_val is not None

        # create on-demand training set
        n_train_batches = self.n_train_samples // self.train_bs
        if n_train_batches == 0:
            raise ValueError(
                f"n_train_samples ({self.n_train_samples}) must be >= train_bs ({self.train_bs})"
            )
        self.n_train_samples = (
            n_train_batches * self.train_bs
        )  # adjust samples count in case of rounding

        self.train_ds = OnDemandDataset(
            self.code,
            self.ebno_dB_train,
            n_train_batches,
            self.train_bs,
            train=True,
            error_space=self.error_space,
            batch_mix=self.batch_mix,
            loss_weights=self.loss_weights,
        )
        # log the resulting SNR distribution by reading back the
        # already-normalized tensors from the dataset
        train_ds = cast(OnDemandDataset, self.train_ds)
        snrs = train_ds.ebno_dB.tolist()
        if len(snrs) == 1:
            log.info(
                f"Created an on-demand training set of {self.n_train_samples} "
                f"true error patterns at Eb/N0={snrs[0]} dB"
            )
        else:
            mix = train_ds.batch_mix.tolist()
            loss_w = train_ds.loss_weights.tolist()
            mix_str = ", ".join(
                f"{snr}dB: mix={m:.3f}, loss_w={lw:.3g}"
                for snr, m, lw in zip(snrs, mix, loss_w)
            )
            log.info(
                f"Created an on-demand training set of {self.n_train_samples} "
                f"true error patterns at Eb/N0 ∈ {{{mix_str}}} (mixed within batches)"
            )

        # create on-demand validation set (always at a single SNR)
        if self.n_val_samples == 0:
            self.n_val_samples = self.n_train_samples // 4
        n_val_batches = self.n_val_samples // self.val_bs
        if n_val_batches == 0:
            raise ValueError(
                f"n_val_samples ({self.n_val_samples}) must be >= val_bs ({self.val_bs})"
            )
        self.n_val_samples = n_val_batches * self.val_bs
        self.val_ds = OnDemandDataset(
            self.code,
            self.ebno_dB_val,
            n_val_batches,
            self.val_bs,
            train=True,
            error_space=self.error_space,
        )
        log.info(
            f"Created an on-demand validation set of {self.n_val_samples} true error patterns at Eb/N0={self.ebno_dB_val} dB"
        )

    def setup(self, stage: str | None = None) -> None:

        if stage in ("validate", "predict"):
            raise NotImplementedError(
                f"SBNDDataModule does not support stage={stage!r}; only 'fit' and 'test' are implemented"
            )

        if stage == "fit":
            if self.on_demand:
                self._setup_on_demand_training()
            else:
                self._setup_dataset_training()
            return

        if stage == "test":

            log.info("Creating the test set(s)...")
            n_test_batches = self.n_test_samples // self.test_bs
            if n_test_batches == 0:
                raise ValueError(
                    f"n_test_samples ({self.n_test_samples}) must be >= test_bs ({self.test_bs})"
                )
            self.n_test_samples = n_test_batches * self.test_bs

            self.test_ds = []
            for ebno_dB in self.ebno_dB_test:
                self.test_ds.append(
                    OnDemandDataset(
                        self.code,
                        ebno_dB,
                        n_test_batches,
                        self.test_bs,
                        train=False,
                        error_space=self.error_space,
                    )
                )
                log.info(
                    f"Created an on-demand test set of {self.n_test_samples} true error patterns at Eb/N0={ebno_dB} dB"
                )

    def on_train_epoch_start(self) -> None:
        # Reshuffle per-dataset index lists in multi-file dataset mode. Called
        # from the main process before the training DataLoader starts iterating;
        # this guarantees workers spawned for the epoch inherit the new state
        # via pickle. Single-file and on-demand training don't need this.
        if isinstance(self.train_ds, MultiDatasetTrainDataset):
            assert self.trainer is not None
            self.train_ds.set_epoch(self.trainer.current_epoch)

    def _train_extra_args(self) -> dict:
        # strip persistent_workers=True for multi-file dataset training
        if isinstance(self.train_ds, MultiDatasetTrainDataset):
            return {
                k: v for k, v in self.extra_args.items() if k != "persistent_workers"
            }
        return self.extra_args

    def train_dataloader(self) -> DataLoader:
        if isinstance(self.train_ds, MultiDatasetTrainDataset) or self.on_demand:
            # batch-producing datasets → automatic batching disabled
            return DataLoader(
                self.train_ds, batch_size=None, **self._train_extra_args()
            )
        # single-file fixed dataset → standard shuffled DataLoader (Lightning
        # auto-wraps with DistributedSampler under DDP)
        return DataLoader(
            self.train_ds, batch_size=self.train_bs, shuffle=True, **self.extra_args
        )

    def val_dataloader(self) -> DataLoader:
        if self.on_demand:
            return DataLoader(self.val_ds, batch_size=None, **self.extra_args)
        else:
            return DataLoader(self.val_ds, batch_size=self.val_bs, **self.extra_args)

    def test_dataloader(self) -> list[DataLoader]:
        test_dls = list(
            map(
                lambda ds: DataLoader(ds, batch_size=None, **self.extra_args),
                self.test_ds,
            )
        )
        return test_dls


if __name__ == "__main__":
    pass
