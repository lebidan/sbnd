import math, torch, numpy as np
import h5py  # type: ignore[import-untyped]
from typing import Iterable, Callable, cast
from omegaconf import OmegaConf, ListConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from lightning import LightningDataModule
from .codes import LinearCode
from scipy.io import loadmat  # type: ignore[import-untyped]
from .utils import get_rank_zero_logger

log = get_rank_zero_logger(__name__)

    
def load_matlab_data(mat_file: str) -> tuple[Tensor, Tensor]:
    """Load received words y and target error patterns e from a MATLAB .mat file."""
    try:
        # v7 mat files or earlier are supported by scipy.io
        matlab_data = loadmat(mat_file, squeeze_me=True)
        y = torch.tensor(matlab_data["y"], dtype=torch.float32)
        e = torch.tensor(matlab_data["e"], dtype=torch.int8)
    except NotImplementedError:
        # but not v7.3 (=HDF5) mat files, for which we need h5py
        f = h5py.File(mat_file, "r")
        y = torch.from_numpy(f["y"][:].astype(np.float32).transpose())
        e = torch.from_numpy(f["e"][:].astype(np.int8).transpose())
    assert e.dtype == torch.int8 and y.dtype == torch.float32
    return y, e


def prepare_data(
    code: LinearCode, y: Tensor, e: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Prepare a batch of training samples for the SBND decoder: create the input pair
    x =(|y|,s) from a received word y and associated binary error pattern e.
    Return the triplet (|y|, s, e), with s in bipolar format (0 -> +1, 1 -> -1)
    Output shape: (bs, n), (bs, m), (bs, n) for an (n,k) code with m parity-check eqs
    """
    ym = torch.abs(y)
    ym = ym / torch.max(ym, dim=1)[0].unsqueeze(
        1
    )  # normalize magnitude to [0,1] within each rx word
    s = code.syndrome(e)
    return ym, (1 - 2 * s).float(), e


def generate_random_training_batch(
    code: LinearCode, ebno_dB: float, bs: int
) -> tuple[Tensor, Tensor]:
    """
    Generate a random batch of bs training samples for the given code and Eb/N0 value (dB)
    Each sample is a tuple (y, e) where:
        - y is the received vector
        - e is the target binary error pattern introduced by the channel
    Transmission of the all-zero codeword is assumed.
    The training samples are only made of rx words with a non-zero syndrome.
    Output shape is (bs, n), (bs, n)
    """
    sigma = 1 / math.sqrt(2 * code.rate * 10 ** (ebno_dB / 10))
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
    """

    def __init__(
        self,
        code: LinearCode,
        ebno_dB: float,
        n_batches: int,
        bs: int,
        train: bool = False,
    ) -> None:
        self.code = code
        self.ebno_dB = ebno_dB
        self.n_batches, self.bs = n_batches, bs
        self.train = train

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        if self.train:
            y, e = generate_random_training_batch(self.code, self.ebno_dB, self.bs)
            return prepare_data(self.code, y, e)
        else:
            y, e = generate_random_test_batch(self.code, self.ebno_dB, self.bs)
            return prepare_data(self.code, y, e)

    def __len__(self) -> int:
        return self.n_batches


class SBNDDataset(Dataset):
    """
    Create a dataset for SBND decoding of the linear code `code`,
    from the pair of tensors (y, e).

    This is essentially a wrapper around Pytorch TensorDataset:
    - https://github.com/pytorch/pytorch/blob/v2.6.0/torch/utils/data/dataset.py#L193

    If `transform` is not None, the specified transform is applied to the data at dataloading.
    """

    def __init__(
        self, code: LinearCode, y: Tensor, e: Tensor, transform: Callable | None = None
    ) -> None:
        assert y.shape[0] == e.shape[0], "y and e must have same number of samples"
        self.code = code
        self.transform = transform
        self.y, self.e = y, e

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        if self.transform is not None:
            return prepare_data(
                self.code, *self.transform(self.y[index], self.e[index])
            )
        else:
            return prepare_data(self.code, self.y[index], self.e[index])

    def __getitems__(self, indices: list[int]) -> list[tuple[Tensor, Tensor, Tensor]]:
        if self.transform is not None:
            y, s, e = prepare_data(
                self.code, *self.transform(self.y[indices], self.e[indices])
            )
            return [(y[i], s[i], e[i]) for i in range(y.size(0))]
        else:
            ym, s, e = prepare_data(self.code, self.y[indices], self.e[indices])
            return [(ym[i], s[i], e[i]) for i in range(ym.size(0))]

    def __len__(self) -> int:
        return self.y.size(0)


class SBNDDataModule(LightningDataModule):
    """
    DataModule for SBND training and testing.

    Can load training/validation data from a .mat file, or generate them on the fly
    at the specified Eb/N0 value.
    Training/validation data never include rx words with a zero syndrome.
    The training/validation data logic depends on the arguments passed to `__init__`.

    If `on_demand` is True (default is False):
        - training and validation will use on-demand data
        - data generation is performed at the Eb/N0 value `ebno_dB_train`
        - `n_train_samples` will be used for training
        - `n_val_samples` will be used for validation
        - `n_train_samples` (resp. `n_val_samples`) will be rounded down to the nearest multiple of `train_bs`
          (resp. `val_bs`) if necessary, as on-demand data generation requires data size to be a multiple of batch size
        - no data augmentation will be applied, even if `transform` is not None

    Else if `train_file` is not None:
        - training will use the first `n_train_samples` samples from `train_file`, or the entire file if `n_train_samples` is 0
        - if `transform` is not None: the specified transform, e.g. cyclic permutation, will be applied to augment the training data
        - if `val_file` is not None: validation will use the first `n_val_samples` samples from `val_file`, or the entire file if `n_val_samples` is 0
        - else (`val_file` is None): a validation set of `n_val_samples` will be created by random split of the training set, whose size `n_train_samples` will be reduced accordingly

    Else if `train_file` is None:
        - a fixed random training set of `n_train_samples` will be created at the SNR value `ebno_dB_train`
        - if `val_file` is not None: validation will use the first `n_val_samples` samples from `val_file`, or the entire file if `n_val_samples` is 0
        - else (`val_file` is None): a fixed random validation set of `n_val_samples` will be created at the SNR value `ebno_dB_train`

    Test data is always generated on the fly, and can include rx words with a zero syndrome.
    It is possible to pass a list of Eb/N0 values to `ebno_dB_test`, in which case different test datasets will be
    generated, one per SNR value, of size `n_test_samples` each.
    Here also `n_test_samples` will be rounded down to the nearest multiple of `test_bs` if required.

    Extra arguments can be passed to pytorch DataLoader, e.g. `num_workers`, through the `extra_args` argument.
    """

    train_ds: Dataset
    val_ds: Dataset
    test_ds: list[Dataset]
    ebno_dB_test: list[float]

    def __init__(
        self,
        code: LinearCode,
        ebno_dB_train: float = 0.0,
        train_file: str | None = None,
        n_train_samples: int = 0,
        train_bs: int = 1024,
        val_file: str | None = None,
        n_val_samples: int = 0,
        val_bs: int = 1024,
        ebno_dB_test: float | Iterable[float] = 0.0,
        n_test_samples: int = 2**20,
        test_bs: int = 1024,
        on_demand: bool = False,
        transform: Callable | None = None,
        extra_args: dict = {},
    ) -> None:
        super().__init__()
        self.code = code
        self.ebno_dB_train = ebno_dB_train
        self.train_file, self.n_train_samples, self.train_bs = (
            train_file,
            n_train_samples,
            train_bs,
        )
        self.val_file, self.n_val_samples, self.val_bs = val_file, n_val_samples, val_bs
        self.ebno_dB_test = ebno_dB_test  # type: ignore[assignment]
        self.n_test_samples, self.test_bs = n_test_samples, test_bs
        self.on_demand = on_demand
        self.transform = transform(code) if transform is not None else None
        self.extra_args = extra_args
        self.save_hyperparameters(
            logger=False
        )  # ensures init params will be stored in ckpt
        # convert ebno_dB_test to python list
        if isinstance(ebno_dB_test, ListConfig):
            self.ebno_dB_test = cast(list[float], OmegaConf.to_object(ebno_dB_test))
        else:
            self.ebno_dB_test = [cast(float, ebno_dB_test)]
        log.info(f"Instantiating an SBNDDataModule for the {code} code")

    def _load_ds(
        self, mat_file: str, n_samples: int, train: bool = False
    ) -> tuple[SBNDDataset, int]:
        y, e = load_matlab_data(mat_file)
        n_samples_in_file = y.shape[0]
        if n_samples == 0:
            n_samples = n_samples_in_file
        else:
            n_samples = min(n_samples, n_samples_in_file)
            y, e = y[:n_samples], e[:n_samples]  # shrink the data if needed
        transform = (
            self.transform if train else None
        )  # apply transforms to training data only
        return SBNDDataset(self.code, y, e, transform), n_samples

    def _random_ds(
        self, ebno_dB: float, n_samples: int, train: bool = False
    ) -> SBNDDataset:
        """Generate a random dataset of n_samples samples at the specified Eb/N0 value, in dB"""
        if train:
            y, e = generate_random_training_batch(self.code, ebno_dB, n_samples)
            transform = self.transform
        else:
            y, e = generate_random_test_batch(self.code, ebno_dB, n_samples)
            transform = None
        return SBNDDataset(self.code, y, e, transform)

    def setup(self, stage: str | None = None) -> None:

        if stage == "fit":

            # special case for on-demand setup
            if self.on_demand:

                n_train_batches = self.n_train_samples // self.train_bs
                self.n_train_samples = (
                    n_train_batches * self.train_bs
                )  # adjust samples count in case of rounding
                assert self.n_train_samples > 0
                self.train_ds = OnDemandDataset(
                    self.code,
                    self.ebno_dB_train,
                    n_train_batches,
                    self.train_bs,
                    train=True,
                )
                log.info(
                    f"Created an on-demand training set of {self.n_train_samples} true error patterns at Eb/N0={self.ebno_dB_train} dB"
                )

                n_val_batches = self.n_val_samples // self.val_bs
                self.n_val_samples = n_val_batches * self.val_bs
                assert self.n_val_samples > 0
                self.val_ds = OnDemandDataset(
                    self.code,
                    self.ebno_dB_train,
                    n_val_batches,
                    self.val_bs,
                    train=True,
                )
                log.info(
                    f"Created an on-demand validation set of {self.n_val_samples} true error patterns at Eb/N0={self.ebno_dB_train} dB"
                )

                return

            # setup training set
            if self.train_file is not None:

                self.train_ds, self.n_train_samples = self._load_ds(
                    self.train_file, self.n_train_samples, train=True
                )
                log.info(
                    f"Loaded {self.n_train_samples} samples from the training set file: {self.train_file}"
                )

                if self.val_file is None and self.n_val_samples > 0:
                    # split training set to create validation set with the requested number of samples
                    assert self.n_val_samples < self.n_train_samples
                    self.n_train_samples -= self.n_val_samples
                    self.train_ds, self.val_ds = random_split(
                        self.train_ds, [self.n_train_samples, self.n_val_samples]
                    )
                    split_ratio = self.n_train_samples / (
                        self.n_train_samples + self.n_val_samples
                    )
                    log.info(
                        f"Created a {100*split_ratio:.0f}%/{100*(1-split_ratio):.0f}% random split with"
                        + f" {self.n_train_samples} samples for training and {self.n_val_samples} samples for validation"
                    )
                    return

            else:

                assert self.n_train_samples > 0
                log.info("Creating a random training set...")
                self.train_ds = self._random_ds(
                    self.ebno_dB_train, self.n_train_samples, train=True
                )
                log.info(
                    f"Created a random training set of {self.n_train_samples} true error patterns at Eb/N0={self.ebno_dB_train} dB"
                )

            # setup validation set
            if self.val_file is not None:

                self.val_ds, self.n_val_samples = self._load_ds(
                    self.val_file, self.n_val_samples, train=False
                )
                log.info(
                    f"Loaded {self.n_val_samples} samples from the validation set file: {self.val_file}"
                )

            else:

                assert self.n_val_samples > 0
                log.info("Creating a random validation set...")
                self.val_ds = self._random_ds(
                    self.ebno_dB_train, self.n_val_samples, train=True
                )
                log.info(
                    f"Created a random validation set of {self.n_val_samples} true error patterns at Eb/N0={self.ebno_dB_train} dB"
                )

        if stage == "test":

            log.info("Creating the test set(s)...")
            n_test_batches = self.n_test_samples // self.test_bs
            self.n_test_samples = n_test_batches * self.test_bs
            assert self.n_test_samples > 0

            self.test_ds = []
            for ebno_dB in self.ebno_dB_test:  # type: ignore[union-attr]
                self.test_ds.append(
                    OnDemandDataset(
                        self.code, ebno_dB, n_test_batches, self.test_bs, train=False
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
