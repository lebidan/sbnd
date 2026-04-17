<p align="center">
  <picture>
    <img alt="SBND" src="https://raw.githubusercontent.com/lebidan/sbnd/main/media/logo.png?raw=true" width=55%>
  </picture>
</p>
<h1 align="center">
Syndrome-Based Neural Decoding
</h1>

<p align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch 2.9+](https://img.shields.io/badge/PyTorch-2.9%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/lebidan/sbnd/blob/main/LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-lightgrey)]()

[Features](#key-features) | [Installation](#installation) | [Quick Start](#quick-start) | [Codes & Decoders](#supported-codes--decoders) | [Configuration](#configuration-guide) | [Project Structure](#project-structure) | [Acknowledgments](#acknowledgments)

**SBND** is a PyTorch/Lightning framework for training and evaluating syndrome-based neural decoders for linear error-correcting codes.

</p>

---

## Features

* **Multiple decoder architectures** — ships with `StackedGRU`, `ECCT`, `CrossMPT`, and `rECCT` (a recurrent ECCT), all sharing a common interface
* **Easy to extend** — add your own architecture using the included [template decoder](https://github.com/lebidan/sbnd/blob/main/src/mocked.py)
* **Hydra configuration** — every aspect of training is configurable via composable YAML files
* **Flexible data pipeline** — train on pre-computed datasets or generate noisy codewords on-the-fly
* **Data augmentation** — leverage code automorphisms to increase training diversity
* **Multi-GPU** — distributed training via PyTorch Lightning DDP
* **Monte Carlo evaluation** — evaluate trained models over configurable Eb/N0 ranges with BER/WER reporting
* **Experiment tracking** — built-in CSV and [Weights & Biases](https://wandb.ai) logging with gradient/weight monitoring

## Installation

To set up a local development environment, we recommend using [uv](https://docs.astral.sh/uv/), which can be installed following [their instructions](https://docs.astral.sh/uv/getting-started/installation/).

### Using uv (recommended)

```
git clone https://github.com/lebidan/sbnd.git
cd sbnd
uv venv
source .venv/bin/activate            # on Windows: .venv\Scripts\activate
uv pip install -e ".[wandb]" --torch-backend=auto
```

The `--torch-backend=auto` command-line option will query the local accelerator and fetch the matching PyTorch wheel automatically (CUDA, CPU, or MPS on Apple Silicon).

Replace `".[wandb]"` with `.` if you do not need [Weights & Biases](https://wandb.ai) integration.

To pin a specific CUDA or torch version, use:

```
uv pip install -e ".[wandb]" torch==2.9.0 --torch-backend=cu128   # force pytorch 2.9 with CUDA 12.8
```

### Using pip

This will install everything with the latest PyTorch and CUDA-compatible wheel:

```
git clone https://github.com/lebidan/sbnd.git
cd sbnd
python -m venv .venv
source .venv/bin/activate            # on Windows: .venv\Scripts\activate
pip install -e ".[wandb]"
```

To pin a specific CUDA variant when the PyPI default does not match your driver:

```
pip install -e ".[wandb]" --extra-index-url https://download.pytorch.org/whl/cu128
```

### Development tools

```
uv pip install black mypy            # or: pip install black mypy
black src/                           # auto-format
mypy src/                            # type checking
```

## Quick Start 🚀

### Verify your installation

Running `sbnd-train` without arguments will automatically execute the default [`dev-test-mocked`](https://github.com/lebidan/sbnd/tree/main/conf/exp/dev-test-mocked.yaml) experiment, which trains a minimal decoder (a single linear layer) for 16 epochs on on-demand generated data:

```
sbnd-train
```

If everything is set up correctly, training should complete in a few minutes.

### Train a model

Training is configured with [Hydra](https://hydra.cc). Each experiment config is located in [`conf/exp/`](https://github.com/lebidan/sbnd/tree/main/conf/exp) and defines a complete training setup: the error-correcting **code**, the **decoder** architecture, the **training data** pipeline, and the **training parameters** (optimizer, LR scheduler, precision, etc.). 

Use the `sbnd-train` CLI to launch a training job, selecting an experiment with `exp=`:

```
sbnd-train exp=ecct-bch-63-45-ml-4m-2dB-aug
```

Any config value can be overridden on the command line:

```
sbnd-train exp=ecct-bch-63-45-ml-4m-2dB-aug gpus=2 cpus=16 max_epochs=64 lr=0.001
```

See the [Configuration Guide](#configuration-guide) for details on how to create your own experiments.

Training artifacts (Hydra config, logs, checkpoints) are saved under `./log/train/runs/YYYY-MM-DD_HH-MM-SS/` to make sure each run is unique. 

Two checkpoints are saved in the `checkpoints/` run subdirectory: `last.ckpt` (model state at the latest epoch) and `<exp-file-name>-<max_epochs>epochs-<wandb-run-name>.ckpt` (best model by validation accuracy). The W&B run name suffix is omitted if W&B is not installed.

### Evaluate a model

`sbnd-test` evaluates a trained checkpoint through Monte Carlo simulation over a range of Eb/N0 values, reporting **Word Error Rate (WER)** and **Bit Error Rate (BER)** (calculated on the message bits) for each SNR point:

```
sbnd-test /path/to/my-model.ckpt --snr_min 1 --snr_max 5 --snr_step 0.5 --num_batches 8192 --batch_size 4096
```

where `/path/to/my-model.ckpt` should have the form `./log/train/runs/YYYY-MM-DD_HH-MM-SS/checkpoints/<exp-file-name>-<max_epochs>epochs-<wandb-run-name>.ckpt`

Results are saved to a CSV file named after the checkpoint under the output directory (default: [`./log/test/`](https://github.com/lebidan/sbnd/tree/main/log/test)). If the file already exists, new SNR points are appended and deduplicated by Eb/N0 value, so you can extend an evaluation incrementally across multiple runs.

| Option | Default | Description |
| --- | --- | --- |
| `--snr_min` / `--snr_max` / `--snr_step` | 0.0 / 5.0 / 1.0 | Eb/N₀ range to simulate (dB) |
| `--batch_size` | 4096 | Test batch size |
| `--num_batches` | 1024 | Number of batches per SNR point |
| `--output` | `./log/test` | Output directory for the results CSV |

## Supported Codes & Decoders

### Codes

A collection of standard BCH, extended BCH, and QC-LDPC codes are shipped in [`data/codes/`](https://github.com/lebidan/sbnd/tree/main/data/codes). Any linear code can be used by providing a MATLAB `.mat` file with the following fields:

| Field | Required | Description |
| --- | --- | --- |
| `n` | ✓ | Code length |
| `k` | ✓ | Message length |
| `G` | ✓ | Generator matrix (k × n) |
| `H` | ✓ | Parity-check matrix (m × n) |
| `dmin` |  | Minimum distance (defaults to 0 if not provided) |
| `name` |  | Code family name (defaults to `"Linear"`) |

### Decoder architectures

SBND ships with four syndrome-based neural decoder architectures:

| Decoder | Class | Source | Reference |
| --- | --- | --- | --- |
| StackedGRU | `sbnd.gru.StackedGRU` | [`gru.py`](https://github.com/lebidan/sbnd/blob/main/src/gru.py) | [Bennatan et al., 2018](https://arxiv.org/abs/1802.04741) |
| ECCT | `sbnd.ecct.ECCT` | [`ecct.py`](https://github.com/lebidan/sbnd/blob/main/src/ecct.py) | [Choukroun & Wolf, 2022](https://arxiv.org/abs/2206.14881) |
| CrossMPT | `sbnd.crossmpt.CrossMPT` | [`crossmpt.py`](https://github.com/lebidan/sbnd/blob/main/src/crossmpt.py) | [Park et al., 2025](https://arxiv.org/abs/2507.01038) |
| rECCT | `sbnd.recct.RECCT` | [`recct.py`](https://github.com/lebidan/sbnd/blob/main/src/recct.py) | [de Boni Rovella, 2024](https://theses.fr/2024ESAE0065) |

The stacked GRU decoder is the straightforward implementation of Bennatan et al.'s (2018) architecture. ECCT and CrossMPT are essentially the verbatim copies of the original implementations, with some little refactoring to speed up attention calculations and a few minor tweaks here and there to slightly improve accuracy. The rECCT decoder is a recurrent implementation of ECCT which can reach comparable performance with fewer parameters (up to 10x less in certain cases). There has been renewed interest recently in recurrent transformers as a parameter-efficient architecture (see, e.g., arxiv papers on looped transformers).

All decoders share the same interface: `forward(ym, s) → logits`, where `ym` is the normalized channel magnitude `|y|/max(|y|)`, `s` is the bipolar syndrome vector, and `logits` is the decoder prediction of the target error pattern. See [`src/mocked.py`](https://github.com/lebidan/sbnd/blob/main/src/mocked.py) for a minimal template to implement your own.

## Configuration Guide 📝

Training is orchestrated by [`SBNDLitModule`](https://github.com/lebidan/sbnd/blob/main/src/model.py), a PyTorch Lightning module wrapper. The SBND decoder architecture to train is passed as a constructor argument to this module. SBND models are trained in a supervised manner, to minimize the average binary cross-entropy between the predicted and target error patterns. The two main metrics monitored during training are **loss** and **accuracy** (the fraction of correctly predicted error patterns).

Training configuration is managed with [Hydra](https://hydra.cc). The base config [`conf/train.yaml`](https://github.com/lebidan/sbnd/blob/main/conf/train.yaml) defines defaults for hardware, logging, callbacks, and path variables such as `codes_dir` (default: `./data/codes`). Experiment configs under [`conf/exp/`](https://github.com/lebidan/sbnd/tree/main/conf/exp) override what they need, following the naming convention `<decoder>-<code>-<data_mode>-<dataset_size>-<snr>[-aug].yaml`. The `dev-test-mocked` experiment is an exception to this convention: it serves as a quick sanity check and is the default when no experiment is specified.

> We recommend starting from the shipped examples in [`conf/exp/`](https://github.com/lebidan/sbnd/tree/main/conf/exp) and adapting them to your needs.

### Code

Specify the error-correcting code by pointing to a `.mat` file (see [Codes](#codes)):

```yaml
code:
  _target_: sbnd.codes.LinearCode
  mat_file: ${codes_dir}/bch.63.45.mat
```

The `codes_dir` variable is defined in [`conf/train.yaml`](https://github.com/lebidan/sbnd/blob/main/conf/train.yaml) and defaults to `./data/codes`.

### Training data

Training and evaluation data is handled by the [`SBNDDataModule`](https://github.com/lebidan/sbnd/blob/main/src/data.py) class. A training sample is a pair `((|y|,s), e)`, where `(|y|,s)` is the decoder input (received LLR magnitude vector and syndrome) and `e` is the target error pattern. SBND models are trained on noisy observations `y = 1 + w` of the all-zero codeword (all-one BPSK modulated codeword), taking advantage of the fact that SBND decoding is agnostic to the transmitted codeword. On the other hand, model evaluation is conducted on randomly generated codewords.

At present, SBND supports three training data strategies:

#### 1. On-demand generation (`on_demand: true`)

Noisy codewords are generated randomly at every training step. The model is exposed to fresh data at each step — no sample is ever repeated. This mode avoids large dataset files and is the simplest way to get started. Note that data augmentation is not applied in this mode, since the data is already unique at every step. The downside is that the model is trained for perfect correction, an unrealistic goal that ultimately hinders WER performance (see our [ICMLCN 2025 paper](https://arxiv.org/abs/2502.10183) for details). In this mode, a training epoch consists of `n_train_samples / train_bs` training steps, or batches.

```yaml
data:
  _target_: sbnd.data.SBNDDataModule
  ebno_dB_train: 2.0
  on_demand: true
  n_train_samples: 1048576
  train_bs: 4096
  n_val_samples: 524288
  val_bs: 4096
```

#### 2. Pre-computed datasets (`on_demand: false` with `train_file` specified)

Load training and validation data from user-supplied `.mat` files. Each file must contain a matrix of received words `y` and a matrix of target binary error patterns `e`. The same fixed dataset is reused at each epoch, which gives total control over the training distribution, making it possible to more closely approach Maximum Likelihood decoding performance with [much fewer samples than with on-demand data](https://arxiv.org/abs/2502.10183). If no `val_file` is provided, a validation set is created by random split from the training set. 

The `data_dir` variable defaults to `./data/datasets` and is defined in [`conf/train.yaml`](https://github.com/lebidan/sbnd/blob/main/conf/train.yaml).

```yaml
data:
  _target_: sbnd.data.SBNDDataModule
  ebno_dB_train: 2.0
  on_demand: false
  train_file: ${data_dir}/bch-63-45/train-ml-4M-2dB.mat
  train_bs: 4096
  val_file: ${data_dir}/bch-63-45/val-ml-512K-2dB.mat
  val_bs: 4096
```

#### 3. Random fixed dataset (`on_demand: false` without `train_file` specified)

A random training set is generated once at the start of training and reused at each epoch. This is a middle ground: no dataset files needed, but the training data remains fixed. A random validation set is also generated unless `val_file` is provided. This configuration usually results in slightly faster convergence than with on-demand data, but with the same downside of training for perfect correction, which is ultimately unachievable.

```yaml
data:
  _target_: sbnd.data.SBNDDataModule
  ebno_dB_train: 2.0
  on_demand: false
  n_train_samples: 1048576
  train_bs: 4096
  n_val_samples: 524288
  val_bs: 4096
```

#### Pre-computed dataset format and download

Pre-computed training datasets are too large to ship with the repository. Most of the training experiments in the [`conf/exp/`](https://github.com/lebidan/sbnd/tree/main/conf/exp) directory can be reproduced with the datasets listed below. Each dataset consists of ML error patterns collected by standard Monte Carlo simulation of an ordered statistics decoder (OSD), and comes as a bundle of training and validation data.

| Code | Dataset description | Size | Link |
| --- | --- | --- | --- |
| BCH(31,21,5) | 4M training + 512K validation samples collected at Eb/N0 = 3 dB | ~1 GB | [Download](https://sdrive.cnrs.fr/s/bKBHagxAwLiNNzn) |
| BCH(63,45,7) | 4M training + 512K validation samples collected at Eb/N0 = 2 dB | ~2.2 GB | [Download](https://sdrive.cnrs.fr/s/wMDN6beY2Gnb7rg) |

Each dataset is stored as a `.mat` and must contain at least the following fields:

| Field | Type | Shape | Description |
| --- | --- | --- | --- |
| `y` | float | (N, n) | Received words (channel output) |
| `e` | float/int8 | (N, n) | Target binary error patterns (ML decoder decisions) |

`N` is the number of samples, `n` is the code length. Both MATLAB v7 and v7.3 (HDF5) formats are supported.

Additional datasets used to produce the results in our [ICMLCN 2025 paper](https://arxiv.org/abs/2502.10183) can be found on the [AI4CODE website](https://ai4code.projects.labsticc.fr/software/), or via their [official DOI](https://doi.org/10.57745/FWE4FB).

#### Data augmentation

For modes 2 and 3 (fixed datasets), data augmentation can be enabled via the `transform` option. This applies random permutations from the code's automorphism group to each batch, effectively multiplying the number of distinct training examples. At present, the following code-specific transforms are available in [`transforms.py`](https://github.com/lebidan/sbnd/blob/main/src/transforms.py):

* [`BCHPerms`](https://github.com/lebidan/sbnd/blob/main/src/transforms.py) — cyclic × Frobenius permutations for BCH codes
* [`QCPerms`](https://github.com/lebidan/sbnd/blob/main/src/transforms.py) — quasi-cyclic shift permutations for QC-LDPC codes (requires the circulant size `Zc`)

```yaml
data:
  transform:
    _partial_: true
    _target_: sbnd.transforms.BCHPerms   # or sbnd.transforms.QCPerms
```

### Model

Select a decoder architecture and configure its hyperparameters:

```yaml
decoder:
  _target_: sbnd.ecct.ECCT
  n_layers: 6
  embed_dim: 128
  n_heads: 8
  attn_dropout: 0.1
  res_dropout: 0.01
  use_fast_attn: true
  compile: true
```

Each architecture exposes its own set of parameters. Refer to the corresponding source file (see [Decoder architectures](#decoder-architectures)) for the full list of options.

### Trainer

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

Similarly, any PyTorch LR scheduler can be used. Two convenience schedulers are provided in [`lr_sched.py`](https://github.com/lebidan/sbnd/blob/main/src/lr_sched.py): `WarmupStableDecayLR` (warmup–stable–decay, recommended; see [Hu et al., 2024](https://arxiv.org/abs/2405.18392)), and  `CosineWarmupLR` (cosine annealing with linear warmup, another classic for transformer models):

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

We recommend using `bf16-mixed` precision for faster training without loss of accuracy, especially with transformer-based models. The only exception is the `StackedGRU` decoder, which we found to require `fp32` precision for both stability and performance. For the full list of supported trainer options, see the [Lightning Trainer documentation](https://lightning.ai/docs/pytorch/stable/common/trainer.html).

**Resuming and continuing training** — two flags are available for working with existing checkpoints:

* `resume=<path>` — resume an interrupted training run from a checkpoint (typically `last.ckpt`). All parameters are restored from the checkpoint and training continues where it left off:

  ```
  sbnd-train exp=ecct-bch-63-45-ml-4m-2dB-aug resume=log/train/runs/.../checkpoints/last.ckpt
  ```
* `continue=<path>` — start a new training run using a pre-trained model as initialization. The model weights are loaded from the checkpoint, but optimizer, scheduler, and other training parameters are taken from the current config:

  ```
  sbnd-train exp=ecct-bch-63-45-ml-4m-2dB-aug continue=log/train/runs/.../checkpoints/best.ckpt lr=0.0001 max_epochs=128
  ```

**Logging** — CSV logging is always enabled. [Weights & Biases](https://wandb.ai) logging is automatically activated when the `wandb` package is installed. To use W&B in offline mode, set `offline: true` in your experiment config or pass `offline=true` on the command line.

## Project Structure

```
sbnd/
├── conf/
│   ├── train.yaml              # Base Hydra config (hardware, logging, callbacks, path variables)
│   └── exp/                    # Experiment configs
├── data/
│   └── codes/                  # Code definition .mat files (G, H, n, k)
├── src/                        # Python package (installed as `sbnd`)
│   ├── codes.py                # LinearCode class
│   ├── data.py                 # SBNDDataModule: datasets and batch generation
│   ├── model.py                # SBNDLitModule: Lightning training wrapper
│   ├── ecct.py                 # ECCT decoder
│   ├── crossmpt.py             # CrossMPT decoder
│   ├── recct.py                # rECCT decoder
│   ├── gru.py                  # StackedGRU decoder
│   ├── mocked.py               # Minimal template decoder
│   ├── transforms.py           # Data augmentation (BCHPerms, QCPerms)
│   ├── lr_sched.py             # LR schedulers (CosineWarmupLR, WarmupStableDecayLR)
│   ├── train.py                # sbnd-train entry point
│   ├── test.py                 # sbnd-test entry point
│   └── utils.py                # Logging utilities
├── pyproject.toml              # Package metadata and dependencies
└── LICENSE                     # MIT License
```

## License

This project is licensed under the [MIT License](https://github.com/lebidan/sbnd/blob/main/LICENSE).

## Contributing

Contributions are welcome. Please open an [issue](https://github.com/lebidan/sbnd/issues) to report bugs or suggest features, and feel free to submit pull requests.

## Acknowledgments

Much of this code was developed within the framework of the [ANR-21 AI4CODE project](https://ai4code.projects.labsticc.fr/).

The following decoder implementations are adapted from their original authors' code:

* **ECCT** — adapted from [yoniLc/ECCT](https://github.com/yoniLc/ECCT) (MIT License), by Y. Choukroun and L. Wolf
* **CrossMPT** — adapted from [iil-postech/crossmpt](https://github.com/iil-postech/crossmpt), by S.-J. Park et al.

This project has greatly benefited from the following open-source software:

* [PyTorch](https://pytorch.org/) — deep learning framework
* [Lightning](https://lightning.ai/) — training infrastructure and multi-GPU support
* [Hydra](https://hydra.cc/) — configuration management

## Citation

If you find this code useful for your own research, please cite:

```bibtex
@misc{lebidan2026sbnd,
      title={SBND: Syndrome-based neural decoding for linear error-correcting codes}, 
      author={Raphaël Le Bidan},
      year={2026},
      url={https://github.com/lebidan/sbnd}, 
}
```