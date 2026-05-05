# Training a model

This document is the reference guide for configuring and running training jobs with `sbnd-train`. It covers the structure of an experiment configuration, the available training data strategies, the model and trainer options, and the conventions around resuming and continuing runs.

**See also:** [README](../README.md#-getting-started) · [Evaluating a model](evaluation.md) · [Extending SBND](extending.md)

## Contents

1. [Overview](#overview)
2. [Code](#code)
3. [Training data](#training-data)
   - [On-demand generation](#on-demand-generation)
   - [Pre-computed datasets](#pre-computed-datasets)
   - [Pre-computed dataset format and download](#pre-computed-dataset-format-and-download)
   - [Mixing on-demand with a fixed dataset](#mixing-on-demand-with-a-fixed-dataset)
   - [Data augmentation](#data-augmentation)
4. [Model](#model)
5. [Trainer](#trainer)
6. [Resuming and continuing training](#resuming-and-continuing-training)
7. [Logging](#logging)
8. [End-of-training test evaluation](#end-of-training-test-evaluation)

## Overview

Training is orchestrated by [`SBNDLitModule`](../src/model.py), a PyTorch Lightning module wrapper around an SBND decoder. The decoder architecture to train is passed as a constructor argument. SBND models are trained in a supervised manner to minimize the average binary cross-entropy between the predicted and target error patterns. The two main metrics monitored during training are the **loss** and the **accuracy** (the fraction of correctly predicted error patterns).

> Despite considerable efforts and experiments with many different loss functions and penalty terms, including [the soft syndrome penalty from Lugosh & Gross](https://arxiv.org/abs/1810.10902) or the [FER loss from Xiao et al.](https://arxiv.org/abs/2105.04118), we have not yet found a suitable loss function that works as consistently well across codes and decoder models as the standard BCE while better optimizing for WER. We'd love to hear from you if you have any suggestions!

Training is configured with [Hydra](https://hydra.cc). The base config [`conf/train.yaml`](../conf/train.yaml) defines defaults for hardware, logging, callbacks, and path variables such as `codes_dir` (default: `./data/codes`). Experiment configs under [`conf/exp/`](../conf/exp) override what they need, following the naming convention `<decoder>-<code>-<data_mode>-<dataset_size>-<snr>[-aug].yaml`. The `dev-test-mocked` experiment serves as a quick sanity check and is the default when no experiment is specified.

> We recommend starting from one of the shipped examples in [`conf/exp/`](../conf/exp) and adapting it to your needs. Each training experiment is paired with a model performance evaluation log file in [`log/test/`](../log/test).

## Code

The error-correcting code is specified by pointing to a `.mat` file. See the [Codes section in the README](../README.md#codes) for the expected file format.

```yaml
code:
  _target_: sbnd.codes.LinearCode
  mat_file: ${codes_dir}/bch.63.45.mat
```

The `codes_dir` variable is defined in [`conf/train.yaml`](../conf/train.yaml) and defaults to `./data/codes`.

## Training data

Training and evaluation data are handled by the [`SBNDDataModule`](../src/data.py) class. A training sample is a pair `((|y|, s), e)`, where `(|y|, s)` is the decoder input (received LLR magnitude vector and bipolar syndrome) and `e` is the target error pattern. SBND models are trained on noisy observations `y = 1 + w` of the all-zero codeword (the all-one BPSK-modulated codeword), exploiting the fact that SBND decoding is agnostic to the transmitted codeword. Model evaluation, in contrast, is conducted on randomly drawn codewords.

> Important note: all the codebase assumes the following binary-to-bipolar BPSK symbol mapping: `0` -> `+1`, `1` -> `-1`.

SBND currently supports two main training data strategies, selected by whether a training dataset file is specified (variable `train_file` set) or not (the default). But both strategies can also be combined to push further model performance.

### On-demand generation

This is the default mode, used when `train_file` is omitted. Noisy codewords are generated randomly at every training step, so the model is exposed to fresh data at each step and no sample is ever repeated. This mode avoids managing large dataset files and is the simplest way to get started. Data augmentation is not applied here, since each batch is already unique. The downside is that the model is effectively trained for perfect correction, an unrealistic objective that ultimately limits WER performance — see our [ICMLCN 2025 paper](https://arxiv.org/abs/2502.10183) for a detailed analysis.

In on-demand mode, a training epoch consists of `n_train_samples / train_bs` steps. The mode requires `ebno_dB_train` and a non-zero `n_train_samples`. If `n_val_samples` is not given, validation defaults to 25% of `n_train_samples`. Both `n_train_samples` and `n_val_samples` are rounded down to the nearest multiple of `train_bs` and `val_bs` respectively.

```yaml
data:
  _target_: sbnd.data.SBNDDataModule
  ebno_dB_train: 2.0
  n_train_samples: 1048576
  train_bs: 4096
  n_val_samples: 524288
  val_bs: 4096
```

#### Mixing several training SNR points

`ebno_dB_train` can also be a list, in which case each row of every training batch is independently drawn at one of the listed SNRs. The optional `ebno_dB_train_weights` argument controls the proportion of samples drawn at each SNR (defaults to uniform; values are normalized to sum to 1). Mixing happens *within* each batch — every gradient step sees a representative spread of easy and hard samples — rather than across batches, which would bias each step toward whichever SNR happened to be drawn for it.

```yaml
data:
  _target_: sbnd.data.SBNDDataModule
  ebno_dB_train: [2.0, 3.0, 4.0]
  ebno_dB_train_weights: [0.6, 0.3, 0.1]   # 60% / 30% / 10%; uniform if omitted
  ebno_dB_val: 3.0                          # required when ebno_dB_train is a list
  n_train_samples: 1048576
  train_bs: 4096
```

> **Pick the weights deliberately — the default uniform is rarely the right choice.** Decoder models often deal well with high-SNR samples: most of them are corrected with little gradient signal. Conversely, low-SNR samples can be too hard to learn. Tune empirically to find the right match for your code and operating conditions.

Validation must always run at a single SNR — an error rate measured on a mix of SNR points is not interpretable — so `ebno_dB_val` selects the validation SNR. When `ebno_dB_train` is a single scalar, `ebno_dB_val` defaults to that value (preserving the previous behavior); when `ebno_dB_train` is a list, `ebno_dB_val` must be specified explicitly.

> Note: the non-zero syndrome filter applied to training samples (a real decoder never operates on zero-syndrome words) shifts the empirical distribution of *kept* samples toward the lower SNR points, since high-SNR rows are more likely to yield a zero syndrome. The weights therefore describe sampling intent before filtering, not the post-filter empirical distribution.

### Pre-computed datasets

When `train_file` is specified, training and validation data are loaded from user-supplied `.mat` dataset files. Each file must contain a matrix of received words `y` and a matrix of target binary error patterns `e`. The same fixed dataset is reused at each epoch, which gives total control over the training distribution. With well-chosen samples, this allows approaching Maximum Likelihood decoding performance with [significantly fewer training samples than on-demand data](https://arxiv.org/abs/2502.10183).

If no `val_file` is provided, a validation set is created by randomly splitting the training set. The default split ratio is 75% / 25%, overridable by setting `n_val_samples` explicitly. The training transform, if any, is applied only to the training subset; validation samples are never augmented.

`n_train_samples` defaults to 0, meaning the entire file is used; set it to a positive value to use only the first N rows. `n_val_samples` behaves identically when loading from `val_file`.

The `data_dir` variable defaults to `./data/datasets` and is defined in [`conf/train.yaml`](../conf/train.yaml).

```yaml
data:
  _target_: sbnd.data.SBNDDataModule
  train_file: ${data_dir}/bch-63-45/train-ml-4M-2dB.mat
  train_bs: 4096
  val_file: ${data_dir}/bch-63-45/val-ml-512K-2dB.mat
  val_bs: 4096
```

**Forbidden combinations.** Setting `val_file` without `train_file` is rejected at construction time (on-demand mode does not load validation data from a file). On-demand mode silently ignores `transform`, since each batch is already unique.

### Pre-computed dataset format and download

Pre-computed training datasets are too large to ship with the repository. Most of the training experiments in [`conf/exp/`](../conf/exp) can be reproduced with the datasets listed below. Each dataset consists of ML error patterns collected by standard Monte Carlo simulation of an ordered statistics decoder (OSD) and ships as a bundle of training and validation data.

| Code | Description | Size | Link |
| --- | --- | --- | --- |
| RM(32,16,8) | 4M training + 512K validation samples at Eb/N0 = 3 dB | ~1 GB | [Download](https://sdrive.cnrs.fr/s/waykmQteWx5RZPn) |
| eBCH(32,16,8) | 4M training + 512K validation samples at Eb/N0 = 3 dB | ~1 GB | [Download](https://sdrive.cnrs.fr/s/fx7kN9s5MwZfi35) |
| BCH(31,21,5) | 4M training + 512K validation samples at Eb/N0 = 3 dB | ~1 GB | [Download](https://sdrive.cnrs.fr/s/bKBHagxAwLiNNzn) |
| BCH(63,45,7) | 4M training + 512K validation samples at Eb/N0 = 2 dB | ~2.2 GB | [Download](https://sdrive.cnrs.fr/s/wMDN6beY2Gnb7rg) |

Each dataset is stored as a `.mat` file containing at least the following fields:

| Field | Type | Shape | Description |
| --- | --- | --- | --- |
| `y` | float | (N, n) | Received words (channel output) |
| `e` | float / int8 | (N, n) | Target binary error patterns (ML decoder decisions) |

`N` is the number of samples and `n` is the code length. Both MATLAB v7 and v7.3 (HDF5) formats are supported.

Additional datasets used to produce the results in our [ICMLCN 2025 paper](https://arxiv.org/abs/2502.10183) are available on the [AI4CODE website](https://ai4code.projects.labsticc.fr/software/) or via their [official DOI](https://doi.org/10.57745/FWE4FB).

### Mixing on-demand with a fixed dataset

The two strategies above can be combined. When both `train_file` and `ebno_dB_train` are set, the DataModule enters a third mode in which each training batch is a random mixture of fixed-dataset samples (with their ML targets) and on-demand samples. This contributes to regularize against dataset memorization and offers the possibility to train models for a very large number of epochs without giving up the ML-target benefit of fixed datasets.

This mode is detected automatically and announced with a warning at startup, so that an unintended combination is hard to miss. If you do not want it, unset either `train_file` or `ebno_dB_train` to switch back to one of the two regular training modes.

```yaml
data:
  _target_: sbnd.data.SBNDDataModule
  train_file: ${data_dir}/bch-63-45/train-ml-4M-2dB.mat
  ebno_dB_train: 2.0                 # or [2.0, 3.0, 4.0] with weights, like in pure on-demand mode
  on_demand_ratio: 0.5               # default; expected fraction of each batch that is on-demand
  train_bs: 4096
```

`on_demand_ratio ∈ [0, 1)` is the only mix-control parameter and reads directly as a percentage. Internally the DataModule sizes the on-demand side as `n_on_demand = round(n_train_samples × ratio / (1 - ratio))` rounded down to a multiple of `train_bs`, then concatenates it with the fixed-dataset side and lets the standard `DataLoader(shuffle=True)` interleave samples per batch. Per-batch composition therefore varies randomly from batch to batch, with a small binomial variance around the target ratio (negligible for typical batch sizes ≥ 1024).

The total epoch length grows accordingly — with `ratio = 0.5`, an epoch is twice as many gradient steps as the fixed-dataset-only baseline, traversing both the full fixed file and an equal-sized on-demand pool.

A few things to keep in mind:
- **Validation is unchanged.** It still uses either the random split of `train_file` or the explicit `val_file`. `ebno_dB_val` has no effect in mixed mode and is ignored with a warning.
- **Augmentation only applies to the fixed dataset samples.** The `transform` is a property of the fixed dataset; on-demand samples are already fresh and never augmented.
- **Multi-SNR mixing on the on-demand half works as in pure on-demand mode.** Pass a list to `ebno_dB_train` and an optional `ebno_dB_train_weights`.
- **Memory cost is essentially that of the fixed file.** The on-demand side is stateless (each DataLoader worker keeps a small per-call buffer; tens of KB).

### Data augmentation

For pre-computed datasets, data augmentation is enabled via the `transform` option. This applies random permutations from the code's automorphism group to each batch, effectively multiplying the number of distinct training examples seen by the model. The same transform classes are reused at evaluation time for [test-time augmentation](evaluation.md#3-test-time-scaling). The classes available in [`src/transforms.py`](../src/transforms.py) are:

* [`BCHPerms`](../src/transforms.py) — cyclic × Frobenius permutations for BCH codes (works for extended BCH too with `is_extended: true`)
* [`QCPerms`](../src/transforms.py) — quasi-cyclic shift permutations for QC-LDPC codes (requires the circulant size `Zc`)
* [`GenericPerms`](../src/transforms.py) — load an arbitrary set of permutations from a `.mat` file (useful for codes whose automorphism group is not directly captured by `BCHPerms` or `QCPerms`)

Example for BCH and QC codes:

```yaml
data:
  transform:
    _partial_: true
    _target_: sbnd.transforms.BCHPerms   # or sbnd.transforms.QCPerms
    is_extended: true                     # for eBCH codes only
```

For `GenericPerms`, the `.mat` file must contain a `perms` field of shape `(n_perms, code.n)` whose rows are valid permutations of `[0, 1, ..., code.n − 1]`. Two example files are shipped under [`data/perms/`](../data/perms) and resolved through the `perms_dir` path variable defined in [`conf/train.yaml`](../conf/train.yaml):

| File | Code | Number of permutations |
| --- | --- | --- |
| `perms.rm.32.mat` | RM(32, k) for any k | 1024 |
| `perms.polar.128.mat` | Polar(128, k) for any k | 4096 |

```yaml
data:
  transform:
    _partial_: true
    _target_: sbnd.transforms.GenericPerms
    mat_file: ${perms_dir}/perms.rm.32.mat
    # num_perms: 256                      # optional — keep only the first N rows; omit (default) to load all
```

A complete worked example using `GenericPerms` with the shipped RM-32 file is provided in [`conf/exp/recct-rm-32-16-ml-4m-3dB-aug.yaml`](../conf/exp/recct-rm-32-16-ml-4m-3dB-aug.yaml).

The same transform classes — including `GenericPerms` and the shipped `.mat` files — are reused at evaluation time for [test-time augmentation](evaluation.md#test-time-augmentation).

To register a new automorphism family, see [Extending SBND](extending.md).

## Model

The decoder architecture and its hyperparameters are configured under the `decoder:` block. Each architecture exposes its own set of parameters; refer to the corresponding source file (see [Decoder architectures](../README.md#decoder-architectures) in the README) for the full list of options.

```yaml
decoder:
  _target_: sbnd.ecct.ECCT
  n_layers: 6
  embed_dim: 128
  n_heads: 8
  attn_dropout: 0.1
  compile: true
```

To implement a new decoder architecture, see [Extending SBND](extending.md).

## Trainer

Any standard PyTorch optimizer can be used via Hydra's `_target_` mechanism. We recommend `AdamW`:

```yaml
max_epochs: 512
lr: 0.001

optimizer:
  _partial_: true
  _target_: torch.optim.AdamW
  lr: ${lr}
  weight_decay: 0.01
```

Similarly, any PyTorch LR scheduler can be used. Two convenience schedulers are provided in [`src/lr_sched.py`](../src/lr_sched.py): `WarmupStableDecayLR` (warmup–stable–decay, recommended; see [Hu et al., 2024](https://arxiv.org/abs/2405.18392)) and `CosineWarmupLR` (cosine annealing with linear warmup, a classic choice for transformer models):

```yaml
lr_scheduler:
  _partial_: true
  _target_: sbnd.lr_sched.WarmupStableDecayLR
  total: ${max_epochs}
  warmup: 10
  decay: 32

trainer:
  precision: bf16-mixed
  gradient_clip_val: 1.0
```

We recommend `bf16-mixed` precision for faster training without loss of accuracy, especially with transformer-based models. The only exception is the `StackedGRU` decoder, which we found to require `fp32` precision for both stability and performance. For the full list of supported trainer options, see the [Lightning Trainer documentation](https://lightning.ai/docs/pytorch/stable/common/trainer.html).

## Resuming and continuing training

Two flags control how an existing checkpoint is consumed by `sbnd-train`:

* **`resume=<path>`** — resume an interrupted training run from a checkpoint (typically `last.ckpt`). All training state is restored from the checkpoint and training continues where it left off:

  ```
  sbnd-train exp=ecct-bch-63-45-ml-4m-2dB-aug \
    +resume=log/train/runs/.../checkpoints/last.ckpt
  ```

* **`continue=<path>`** — start a new training run using a pre-trained model as initialization. The model weights are loaded from the checkpoint, but the optimizer, scheduler, and other training parameters are taken from the current config:

  ```
  sbnd-train exp=ecct-bch-63-45-ml-4m-2dB-aug \
    +continue=log/train/runs/.../checkpoints/best.ckpt \
    lr=0.0001 max_epochs=128
  ```

## Logging

CSV logging is always enabled. [Weights & Biases](https://wandb.ai) logging is automatically activated when the `wandb` package is installed. To use W&B in offline mode, set `offline: true` in the experiment config or pass `offline=true` on the command line.

## End-of-training test evaluation

`sbnd-train` automatically runs a final test evaluation on the best checkpoint at the end of every training run (`trainer.test` is called after `trainer.fit`). Test data is generated on the fly at each Eb/N0 value listed in `ebno_dB_test`, and the following metrics are reported and logged for each SNR point:

* `test/loss` — cross-entropy loss on the test set
* `test/acc` — fraction of correctly predicted error patterns (word accuracy)
* `test/err` — word error rate (WER = 1 − acc)

The test SNR range, sample count, and batch size are configured under the `data:` block:

```yaml
data:
  ebno_dB_test: [2.0, 3.0, 4.0]   # one test set per SNR value
  n_test_samples: 2097152          # 2M samples per SNR point (default)
  test_bs: 4096
```

`n_test_samples` is rounded down to the nearest multiple of `test_bs`. In addition, the `PeriodicTest` callback runs a lightweight interim test evaluation every `every_n_epochs` epochs (default: 50) during training, logging results under the `periodic_test/` namespace. This allows monitoring test-set progress without waiting for the full training run to complete. The interval can be changed in the experiment config:

```yaml
periodic_test_cb:
  _target_: sbnd.train.PeriodicTest
  every_n_epochs: 100   # or 0 to disable
```

For a more comprehensive evaluation, including bit-error rate and the test-time scaling options, use the dedicated `sbnd-test` command — see [Evaluating a model](evaluation.md).
