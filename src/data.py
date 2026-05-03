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


def build_snr_dist(
    ebno_dB: float | list[float], weights: list[float] | None
) -> tuple[Tensor, Tensor]:
    """Normalize a `(ebno_dB, weights)` pair into a `(Tensor, Tensor)` of equal
    length k ≥ 1, where weights sum to 1. Uniform weights are used when
    `weights is None`. Validates length match, non-negativity, and positive sum."""
    if isinstance(ebno_dB, (list, tuple)):
        ebno_list = [float(v) for v in ebno_dB]
    else:
        ebno_list = [float(ebno_dB)]
    if len(ebno_list) == 0:
        raise ValueError("ebno_dB must contain at least one value")
    ebno_t = torch.tensor(ebno_list, dtype=torch.float32)
    if weights is None:
        w_t = torch.full((len(ebno_list),), 1.0 / len(ebno_list), dtype=torch.float32)
    else:
        if len(weights) != len(ebno_list):
            raise ValueError(
                f"weights length ({len(weights)}) must match ebno_dB length ({len(ebno_list)})"
            )
        w_t = torch.tensor([float(x) for x in weights], dtype=torch.float32)
        if torch.any(w_t < 0):
            raise ValueError(f"weights must be non-negative; got {weights}")
        w_sum = w_t.sum()
        if w_sum.item() <= 0:
            raise ValueError(f"weights must have a positive sum; got {weights}")
        w_t = w_t / w_sum
    return ebno_t, w_t


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


def generate_random_training_batch(
    code: LinearCode, ebno_dB: Tensor, weights: Tensor, bs: int
) -> tuple[Tensor, Tensor]:
    """
    Generate a random batch of bs training samples for the given code and Eb/N0
    distribution defined by (`ebno_dB`, `weights`).
    Each sample is a tuple (y, e) where:
        - y is the received vector
        - e is the target binary error pattern introduced by the channel
    Transmission of the all-zero codeword is assumed.
    The training samples are only made of rx words with a non-zero syndrome.

    `ebno_dB` is a 1D tensor listing the SNR points to sample from and `weights`
    is a 1D tensor of same length summing to 1 giving the probability mass at each
    point. Each of the bs rows independently draws its SNR from this categorical
    distribution at every call. The SNR list can reduce to a single value, in
    which case the same SNR is used for all training samples in the batch.

    Note that the non-zero syndrome filter shifts the empirical distribution of
    kept samples toward the lower-SNR points, since high-SNR samples are more likely
    to yield zero syndromes.

    Output shape is (bs, n), (bs, n)
    """
    assert (
        ebno_dB.dim() == 1 and weights.dim() == 1 and ebno_dB.numel() == weights.numel()
    )
    row_idx = torch.multinomial(weights, bs, replacement=True)
    ebno_per_row = ebno_dB[row_idx]
    sigma = (1 / torch.sqrt(2 * code.rate * 10 ** (ebno_per_row / 10))).unsqueeze(1)
    n_samples = 0
    y = torch.empty(0, code.n, dtype=torch.float32)
    e = torch.empty(0, code.n, dtype=torch.int8)
    while n_samples < bs:
        y_ = 1 + sigma * torch.randn(bs, code.n)
        e_ = (y_ < 0).to(torch.int8)
        s_ = code.syndrome(e_)
        nz_synd_idx = torch.any(s_, dim=1).nonzero().squeeze(1)
        n_samples_to_add = min(bs - n_samples, nz_synd_idx.shape[0])
        if n_samples_to_add > 0:
            nz_rows = nz_synd_idx[:n_samples_to_add]
            y = torch.cat((y, y_[nz_rows]))
            e = torch.cat((e, e_[nz_rows]))
            n_samples += n_samples_to_add
    return y, e


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
    distribution to sample from. The optional `weights` argument controls the
    proportion of samples drawn at each SNR (defaults to uniform; values are
    normalized to sum to 1).

    Test mode requires a single `ebno_dB` value.

    Internally `ebno_dB` and `weights` are always stored as 1D tensors of
    same length ≥ 1, so the per-row sampling logic in `generate_random_training_batch`
    handles both single- and multi-SNR cases uniformly.
    """

    ebno_dB: Tensor
    weights: Tensor

    def __init__(
        self,
        code: LinearCode,
        ebno_dB: float | list[float],
        n_batches: int,
        bs: int,
        train: bool = False,
        error_space: str = "codeword",
        weights: list[float] | None = None,
    ) -> None:
        self.code = code
        self.n_batches, self.bs = n_batches, bs
        self.train = train
        self.error_space = error_space

        self.ebno_dB, self.weights = build_snr_dist(ebno_dB, weights)
        if not train and self.ebno_dB.numel() != 1:
            raise ValueError(
                f"OnDemandDataset in test mode requires a single ebno_dB; got {self.ebno_dB.tolist()}"
            )

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        if self.train:
            y, e = generate_random_training_batch(
                self.code, self.ebno_dB, self.weights, self.bs
            )
        else:
            # test mode is constrained to a single SNR by __init__
            y, e = generate_random_test_batch(
                self.code, float(self.ebno_dB[0].item()), self.bs
            )
        return prepare_data(self.code, y, e, self.error_space)

    def __len__(self) -> int:
        return self.n_batches


class OnDemandSampleDataset(Dataset):
    """
    Sample-level on-demand dataset. Each `__getitem__` returns one fresh
    training sample `(ym, s, e)` drawn from the SNR distribution defined by
    `(ebno_dB, weights)`. Length is fixed at construction. Like
    `OnDemandDataset`, all samples have non-zero syndromes.

    Used to build a `MixedTrainDataset` together with a fixed `SBNDDataset`,
    so that a single map-style `DataLoader` can mix both data sources at the
    sample level via standard random shuffling.

    For efficiency, each worker maintains an internal buffer of size
    `chunk_size`: requests are served from the buffer, and the buffer is
    refilled in one vectorized call to `generate_random_training_batch` when
    empty. This amortizes the per-call overhead of rejection sampling. The
    buffer is per-DataLoader-worker and reseeded by `seed_everything(workers=
    True)` along with the global RNG.
    """

    ebno_dB: Tensor
    weights: Tensor

    def __init__(
        self,
        code: LinearCode,
        ebno_dB: float | list[float],
        n_samples: int,
        error_space: str = "codeword",
        weights: list[float] | None = None,
        chunk_size: int = 256,
    ) -> None:
        self.code = code
        self.n_samples = n_samples
        self.error_space = error_space
        self.chunk_size = chunk_size
        self.ebno_dB, self.weights = build_snr_dist(ebno_dB, weights)
        # `_buf` holds chunked, already-prepared samples ready to be served.
        # Lazy-init avoids spending time before the first `__getitem__`.
        self._buf: list[tuple[Tensor, Tensor, Tensor]] = []

    def _refill(self) -> None:
        y, e = generate_random_training_batch(
            self.code, self.ebno_dB, self.weights, self.chunk_size
        )
        ym, s, e = prepare_data(self.code, y, e, self.error_space)
        self._buf = [(ym[i], s[i], e[i]) for i in range(self.chunk_size)]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        # `idx` is ignored — every call yields a fresh sample
        if not self._buf:
            self._refill()
        return self._buf.pop()

    def __getitems__(self, indices: list[int]) -> list[tuple[Tensor, Tensor, Tensor]]:
        # batched fast path: generate exactly `len(indices)` samples in one call
        n = len(indices)
        y, e = generate_random_training_batch(self.code, self.ebno_dB, self.weights, n)
        ym, s, e = prepare_data(self.code, y, e, self.error_space)
        return [(ym[i], s[i], e[i]) for i in range(n)]

    def __len__(self) -> int:
        return self.n_samples


class SBNDDataset(Dataset):
    """
    Create a dataset for SBND decoding of the linear code `code`,
    from the pair of tensors (y, e).

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
    ) -> None:
        assert y.shape[0] == e.shape[0], "y and e must have same number of samples"
        self.code = code
        self.transform = transform
        self.error_space = error_space
        self.y, self.e = y, e

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        # `prepare_data` expects a (bs, n) batch; wrap the single row in a
        # 1-row batch and squeeze the result back. (DataLoader normally hits
        # `__getitems__` instead, but single-sample access matters when this
        # dataset is wrapped in a higher-level structure such as
        # `MixedTrainDataset`.)
        y_row = self.y[index].unsqueeze(0)
        e_row = self.e[index].unsqueeze(0)
        if self.transform is not None:
            y_row, e_row = self.transform(y_row, e_row)
        ym, s, e = prepare_data(self.code, y_row, e_row, self.error_space)
        return ym.squeeze(0), s.squeeze(0), e.squeeze(0)

    def __getitems__(self, indices: list[int]) -> list[tuple[Tensor, Tensor, Tensor]]:
        if self.transform is not None:
            y_t, e_t = self.transform(self.y[indices], self.e[indices])
            y, s, e = prepare_data(self.code, y_t, e_t, self.error_space)
            return [(y[i], s[i], e[i]) for i in range(y.size(0))]
        else:
            ym, s, e = prepare_data(
                self.code, self.y[indices], self.e[indices], self.error_space
            )
            return [(ym[i], s[i], e[i]) for i in range(ym.size(0))]

    def __len__(self) -> int:
        return self.y.size(0)


class MixedTrainDataset(Dataset):
    """
    Concatenation of a fixed `SBNDDataset` (loaded from a `.mat` file) and an
    `OnDemandSampleDataset` (fresh samples drawn from a configurable SNR
    distribution).

    Functionally equivalent to `torch.utils.data.ConcatDataset`, with one
    extra: `__getitems__` is forwarded to the children so DataLoader can keep
    using batched fetching (which is meaningfully faster than per-sample
    `__getitem__` calls). Indices in [0, len(fixed)) map to the fixed side;
    indices ≥ len(fixed) map to the on-demand side.
    """

    def __init__(self, fixed: Dataset, on_demand: OnDemandSampleDataset) -> None:
        # `fixed` is typed as `Dataset` so it accepts both `SBNDDataset` and a
        # `Subset[SBNDDataset]` (used when validation is created by split).
        # Both forward `__getitems__`, which is required by the fast path below.
        self.fixed = fixed
        self.on_demand = on_demand
        self.n_fixed = len(fixed)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return self.n_fixed + len(self.on_demand)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        if idx < self.n_fixed:
            return self.fixed[idx]
        return self.on_demand[idx - self.n_fixed]

    def __getitems__(self, indices: list[int]) -> list[tuple[Tensor, Tensor, Tensor]]:
        # split incoming indices by source while remembering positions, fetch
        # each side in one batched call, then re-assemble in original order
        items: list[tuple[Tensor, Tensor, Tensor] | None] = [None] * len(indices)
        fixed_pos: list[int] = []
        fixed_idx: list[int] = []
        od_pos: list[int] = []
        od_idx: list[int] = []
        for pos, idx in enumerate(indices):
            if idx < self.n_fixed:
                fixed_pos.append(pos)
                fixed_idx.append(idx)
            else:
                od_pos.append(pos)
                od_idx.append(idx - self.n_fixed)
        if fixed_idx:
            for pos, item in zip(fixed_pos, self.fixed.__getitems__(fixed_idx)):  # type: ignore[attr-defined]
                items[pos] = item
        if od_idx:
            for pos, item in zip(od_pos, self.on_demand.__getitems__(od_idx)):
                items[pos] = item
        return cast(list[tuple[Tensor, Tensor, Tensor]], items)


class SBNDDataModule(LightningDataModule):
    """
    DataModule for SBND training and testing.

    Can load training/validation data from a .mat file, generate them on the
    fly at one or several Eb/N0 values, or mix the two within each batch.
    Training/validation data never include rx words with a zero syndrome.
    The training/validation data logic depends on the arguments passed to
    `__init__`, which select one of three modes:

    1. Fixed-dataset (`train_file` set, `ebno_dB_train` unset):
        - training will use the first `n_train_samples` samples from `train_file`, or the entire file if `n_train_samples` is 0 (default)
        - if `transform` is not None: the specified transform, e.g. cyclic permutation, will be applied to augment the training data
        - if `val_file` is not None: validation will use the first `n_val_samples` samples from `val_file`, or the entire file if `n_val_samples` is 0 (default)
        - else (`val_file` is None): a validation set of `n_val_samples` will be created by random split of the training set, whose size `n_train_samples` will be reduced accordingly
          Default split ratio is 75%/25% unless `n_val_samples` is explicitly specified

    2. On-demand (`train_file` unset, `ebno_dB_train` set):
        - training and validation will use on-demand data
        - data generation is performed at the Eb/N0 value(s) given by `ebno_dB_train`
        - `n_train_samples` will be used for training (must be specified explicitely)
        - `n_val_samples` will be used for validation (default value = 25% of `n_train_samples`)
        - `n_train_samples` (resp. `n_val_samples`) will be rounded down to the nearest multiple of `train_bs`
          (resp. `val_bs`) if necessary, as on-demand data generation requires data size to be a multiple of batch size
        - no data augmentation will be applied, even if `transform` is not None

    3. Mixed (both `train_file` and `ebno_dB_train` set): the fixed dataset is
       augmented with fresh on-demand samples, and each training batch is a
       random mixture of the two via standard `DataLoader(shuffle=True)` over a
       map-style concatenation. The mix is controlled by a single parameter:
        - `on_demand_ratio` ∈ [0, 1) — expected fraction of each batch drawn
          from the on-demand side (default 0.5). The on-demand side is sized
          as `round(n_train_samples * ratio / (1 - ratio))`, rounded down to a
          multiple of `train_bs`. A loud warning is logged at startup so an
          unintended combination is hard to miss.
        - validation is unchanged from fixed-dataset mode (random split of
          `train_file` or the explicit `val_file`); `ebno_dB_val` is ignored.
        - augmentation only applies to the fixed half — on-demand samples are
          already fresh and never augmented.
        - multi-SNR sampling on the on-demand half (via a list `ebno_dB_train`
          and optional `ebno_dB_train_weights`) works just like in pure
          on-demand mode.

    On-demand training (modes 2 and 3) supports mixing several SNR points
    within each batch: pass a list to `ebno_dB_train` (instead of a single
    float). The optional `ebno_dB_train_weights` argument gives the proportion
    of samples drawn at each SNR (defaults to uniform; values are normalized
    to sum to 1). Each sample's SNR is drawn i.i.d. from the resulting
    categorical distribution at every batch.

    Validation must always run at a single SNR to be meaningful. In pure
    on-demand mode, `ebno_dB_val` selects it: when `ebno_dB_train` is a single
    float, `ebno_dB_val` defaults to that value (backward compatible); when
    `ebno_dB_train` is a list, `ebno_dB_val` must be specified explicitly. In
    fixed-dataset and mixed modes, validation comes from the file (split or
    `val_file`) and `ebno_dB_val` has no effect.

    Test data is always generated on the fly, and can include rx words with a zero syndrome.
    It is possible to pass a list of Eb/N0 values to `ebno_dB_test`, in which case different test datasets will be
    generated, one per SNR value, of size `n_test_samples` each.
    Here also `n_test_samples` will be rounded down to the nearest multiple of `test_bs` if required.

    Extra arguments can be passed to pytorch DataLoader, e.g. `num_workers`, through the `extra_args` argument.

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
    ebno_dB_train_weights: list[float] | None
    ebno_dB_val: float | None
    ebno_dB_test: list[float]
    on_demand_ratio: float

    def __init__(
        self,
        code: LinearCode,
        ebno_dB_train: float | Iterable[float] | None = None,
        ebno_dB_train_weights: Iterable[float] | None = None,
        train_file: str | None = None,
        n_train_samples: int = 0,
        train_bs: int = 1024,
        val_file: str | None = None,
        n_val_samples: int = 0,
        val_bs: int = 1024,
        ebno_dB_val: float | None = None,
        ebno_dB_test: float | Iterable[float] = 0.0,
        n_test_samples: int = 2**20,
        test_bs: int = 1024,
        on_demand_ratio: float | None = None,
        transform: Callable | None = None,
        error_space: str = "codeword",
        extra_args: dict | None = None,
    ) -> None:
        super().__init__()

        self.code = code
        log.info(f"Instantiating an SBNDDataModule for the {code} code")

        # determine training mode:
        #   - train_file is None                       → pure on-demand
        #   - train_file is set, ebno_dB_train is None → pure fixed-dataset
        #   - both set                                 → mixed (file + on-demand)
        self.on_demand = train_file is None
        self.mixed = train_file is not None and ebno_dB_train is not None

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
            if on_demand_ratio is not None:
                log.warning(
                    "on_demand_ratio is set but training is purely on-demand "
                    "(no fixed dataset to mix with); the value will be ignored"
                )
        elif self.mixed:
            # MIXED mode is detected when both train_file and ebno_dB_train are
            # set. We log it explicitly so the user can stop the run if this is
            # not what they wanted.
            ratio = 0.5 if on_demand_ratio is None else float(on_demand_ratio)
            if not (0.0 <= ratio < 1.0):
                raise ValueError(f"on_demand_ratio must be in [0, 1); got {ratio}")
            self.on_demand_ratio = ratio
            log.warning(
                "Mixed training mode detected: each batch will mix samples from "
                f"the fixed dataset {train_file!r} with fresh on-demand samples "
                f"at Eb/N0={ebno_dB_train} (on_demand_ratio={ratio:.2f}, "
                f"{'default' if on_demand_ratio is None else 'user-set'}). "
                "If this is not what you intended, stop and unset either "
                "train_file or ebno_dB_train."
            )
            if ebno_dB_val is not None:
                log.warning(
                    "ebno_dB_val is set but mixed-mode validation uses the "
                    "fixed-dataset file split or val_file; the value will be ignored"
                )
        else:
            # pure fixed-dataset mode
            log.info(f"Training uses the fixed dataset {train_file}")
            if ebno_dB_val is not None:
                log.warning(
                    "ebno_dB_val is set but training uses a fixed dataset file; "
                    "the value will be ignored"
                )
            if ebno_dB_train_weights is not None:
                log.warning(
                    "ebno_dB_train_weights is set but training uses a fixed "
                    "dataset file with no on-demand mixing; the value will be ignored"
                )
            if on_demand_ratio is not None:
                log.warning(
                    "on_demand_ratio is set but training uses a fixed dataset "
                    "file with no on-demand mixing; the value will be ignored"
                )

        # store non-SNR arguments
        self.train_file, self.n_train_samples, self.train_bs = (
            train_file,
            n_train_samples,
            train_bs,
        )
        self.val_file, self.n_val_samples, self.val_bs = val_file, n_val_samples, val_bs
        self.n_test_samples, self.test_bs = n_test_samples, test_bs
        self.transform = transform(code) if transform is not None else None
        self.error_space = error_space
        self.extra_args = extra_args if extra_args is not None else {}
        self.save_hyperparameters(
            logger=False
        )  # snapshots __init__ args for ckpt; safe to mutate self attrs below

        # coerce list-shaped Hydra args (ListConfig/list/tuple/scalar) into
        # plain `list[float]`. For training, `OnDemandDataset` is the single
        # source of truth for SNR/weights validation and for normalizing weights
        # to sum to 1; we just pass these through to it.
        self.ebno_dB_test = to_float_list(ebno_dB_test) or [0.0]
        self.ebno_dB_train = to_float_list(ebno_dB_train)
        self.ebno_dB_train_weights = to_float_list(ebno_dB_train_weights)

        # resolve the validation SNR — single value required to be meaningful.
        # Only used in pure on-demand mode (mixed and pure-fixed modes derive
        # validation from the file). In on-demand mode, defaults to the training
        # SNR if that is a single value (backward compatible); must be specified
        # explicitly when training mixes several SNRs.
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

        # ensure on_demand_ratio attribute is always defined; set to 0 in modes
        # where there is no mixing happening
        if not self.mixed:
            self.on_demand_ratio = 0.0

    def _load_ds(
        self, mat_file: str, n_samples: int, train: bool = False
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
            SBNDDataset(self.code, y, e, transform, self.error_space),
            n_samples,
        )

    def setup(self, stage: str | None = None) -> None:

        if stage in ("validate", "predict"):
            raise NotImplementedError(
                f"SBNDDataModule does not support stage={stage!r}; only 'fit' and 'test' are implemented"
            )

        if stage == "fit":

            # setup training set
            if self.train_file is not None:

                loaded_ds, self.n_train_samples = self._load_ds(
                    self.train_file, self.n_train_samples, train=True
                )
                self.train_ds = loaded_ds
                log.info(
                    f"Loaded {self.n_train_samples} samples from the training set file: {self.train_file}"
                )

                if self.val_file is not None:

                    # load validation set from user-specified file
                    self.val_ds, self.n_val_samples = self._load_ds(
                        self.val_file, self.n_val_samples, train=False
                    )
                    log.info(
                        f"Loaded {self.n_val_samples} samples from the validation set file: {self.val_file}"
                    )

                else:

                    if self.n_val_samples == 0:
                        # validation set size defaults to 25% of training set
                        self.n_val_samples = self.n_train_samples // 4

                    if self.n_val_samples >= self.n_train_samples:
                        raise ValueError(
                            f"n_val_samples ({self.n_val_samples}) must be < the number of loaded "
                            f"training samples ({self.n_train_samples})"
                        )
                    self.n_train_samples -= self.n_val_samples
                    train_subset, val_subset = random_split(
                        loaded_ds, [self.n_train_samples, self.n_val_samples]
                    )
                    # build a sibling dataset that shares the same (y, e) tensor storage
                    # but has no transform, so validation samples are not augmented
                    val_base = SBNDDataset(
                        self.code,
                        loaded_ds.y,
                        loaded_ds.e,
                        transform=None,
                        error_space=self.error_space,
                    )
                    self.val_ds = Subset(val_base, list(val_subset.indices))
                    self.train_ds = train_subset
                    split_ratio = self.n_train_samples / (
                        self.n_train_samples + self.n_val_samples
                    )
                    log.info(
                        f"Created a {100*split_ratio:.0f}%/{100*(1-split_ratio):.0f}% random split with"
                        + f" {self.n_train_samples} samples for training and {self.n_val_samples} samples for validation"
                    )

                # in MIXED mode, augment the fixed training set with an
                # `OnDemandSampleDataset` of size derived from `on_demand_ratio`
                # (rounded down to a multiple of `train_bs` for clean batching).
                # Validation is left untouched — it stays purely fixed-dataset.
                if self.mixed:
                    assert self.ebno_dB_train is not None
                    ratio = self.on_demand_ratio
                    n_on_demand = round(self.n_train_samples * ratio / (1 - ratio))
                    n_on_demand = (n_on_demand // self.train_bs) * self.train_bs
                    on_demand_ds = OnDemandSampleDataset(
                        self.code,
                        self.ebno_dB_train,
                        n_samples=n_on_demand,
                        error_space=self.error_space,
                        weights=self.ebno_dB_train_weights,
                    )
                    self.train_ds = MixedTrainDataset(self.train_ds, on_demand_ds)
                    snrs = on_demand_ds.ebno_dB.tolist()
                    weights = on_demand_ds.weights.tolist()
                    if len(snrs) == 1:
                        snr_str = f"Eb/N0={snrs[0]} dB"
                    else:
                        mix_str = ", ".join(
                            f"{s}dB: {w:.3f}" for s, w in zip(snrs, weights)
                        )
                        snr_str = f"Eb/N0 ∈ {{{mix_str}}}"
                    log.info(
                        f"Augmented training set with {n_on_demand} on-demand "
                        f"samples ({snr_str}); each batch is now ~{100*ratio:.0f}% "
                        f"on-demand in expectation"
                    )

                return

            else:

                # defaults to on-demand setup
                assert self.on_demand
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
                    weights=self.ebno_dB_train_weights,
                )
                # log the resulting SNR distribution by reading back the
                # already-normalized tensors from the dataset
                snrs = self.train_ds.ebno_dB.tolist()
                if len(snrs) == 1:
                    log.info(
                        f"Created an on-demand training set of {self.n_train_samples} "
                        f"true error patterns at Eb/N0={snrs[0]} dB"
                    )
                else:
                    weights = self.train_ds.weights.tolist()
                    mix_str = ", ".join(
                        f"{snr}dB: {w:.3f}" for snr, w in zip(snrs, weights)
                    )
                    log.info(
                        f"Created an on-demand training set of {self.n_train_samples} "
                        f"true error patterns at Eb/N0 ∈ {{{mix_str}}} (mixed within batches)"
                    )

                # create on-demand validation set (always at a single SNR)
                if self.n_val_samples == 0:
                    # validation set size defaults to 25% of training set
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
            for ebno_dB in self.ebno_dB_test:  # type: ignore[union-attr]
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

    def train_dataloader(self) -> DataLoader:
        if self.on_demand:
            return DataLoader(self.train_ds, batch_size=None, **self.extra_args)
        else:
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
