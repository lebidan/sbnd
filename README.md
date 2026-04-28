<p align="center">
  <picture>
    <img alt="SBND" src="https://raw.githubusercontent.com/lebidan/sbnd/main/media/logo.png?raw=true" width=55%>
  </picture>
</p>
<h1 align="center">
Syndrome-Based Neural Decoding
</h1>

<p align="center">
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+"/></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.9%2B-orange" alt="PyTorch 2.9+"/></a>
<a href="https://github.com/lebidan/sbnd/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"/></a>
<img src="https://img.shields.io/badge/version-0.1.0-lightgrey" alt="Version"/>
</p>
<p align="center">
<a href="#-why-sbnd">Overview</a> |
<a href="#-features">Features</a> |
<a href="#-installation">Installation</a> |
<a href="#-getting-started">Getting Started</a> |
<a href="#-supported-codes--decoders">Codes &amp; Decoders</a> |
<a href="#-configuration-guide">Configuration</a> |
<a href="#-project-structure">Structure</a> |
<a href="#-contributing">Contributing</a> |
<a href="#-acknowledgments">Acknowledgments</a>
</p>

**`SBND`** is a PyTorch/Lightning framework for training and evaluating syndrome-based neural decoders for linear error-correcting codes.

---

## 👀 Why SBND? 

Syndrome-based neural decoding is a promising approach for soft-decision decoding of short, high-rate codes, but the field is still wide open. Performance lags behind classical decoders like OSD or Chase-2, scaling laws are poorly understood, and more parameter-efficient architectures are yet to be found.

`SBND` is built for researchers who want to close that gap. It ships with multiple architectures, reproducible baselines, and a clean training infrastructure — everything you need to run experiments, test new ideas, and push neural decoders further than they've been before. 

<b> ⭐ Performance highlights ⭐</b>

<details><summary>Decoding the (63,45,7) BCH code</summary>

<img alt="BCH(63,45,7) performance" src="https://raw.githubusercontent.com/lebidan/sbnd/main/media/fer_63_45.png?raw=true" width=90%>

- Training the original ECCT with SBND brings ~two-order of magnitude FER improvement
- Same or better performance with half the number of parameters when switching to our recurrent ECCT model
- Performance is within 0.2 dB of MLD and matches Chase-2 decoding with 64 test patterns

Configuration files for the above experiments: [original/improved ECCT training](https://github.com/lebidan/sbnd/blob/main/conf/exp/ecct-bch-63-45-on-demand-2dB.yaml), [rECCT training](https://github.com/lebidan/sbnd/blob/main/conf/exp/recct-bch-63-45-ml-4m-2dB-aug.yaml)

</details>

<details><summary>Decoding the (32,16,8) extended BCH code</summary>

<img alt="eBCH(32,16,8) performance" src="https://raw.githubusercontent.com/lebidan/sbnd/main/media/fer_32_16.png?raw=true" width=90%>

- FER performance within 0.2 dB of MLD and comparable to Chase-2 decoding with 64 test patterns
- Outperforms the original ECCT and CrossMPT decoders with 8x fewer parameters

Note: The comparison between results for the (31,16,7) and (32,16,8) codes is reasonable as both codes have very close MLD performance down to FER = 1E-4. The extended code progressively takes over at high SNRs. Compare with the results in Table 3 and Fig. 11 from [the CrossMPT ICLR 2025 paper](https://openreview.net/forum?id=gFvRRCnQvX).

Configuration file to reproduce the rECCT results: [here](https://github.com/lebidan/sbnd/blob/main/conf/exp/recct-ebch-32-16-ml-16m-3dB.yaml)

</details>

<details><summary>Decoding the (96,48,10) quasi-cyclic LDPC</summary>

<img alt="QC-LDPC(96,48,10) performance" src="https://raw.githubusercontent.com/lebidan/sbnd/main/media/fer_96_48.png?raw=true" width=90%>

- High-SNR FER performance within 1.0 dB or less of MLD (still much room for improvement)
- Matches or outperforms BP with 100 iterations

This very nice and strong short quasi-cyclic LDPC code was designed at [RPTU](https://rptu.de/channel-codes/channel-codes-database/more-ldpc-codes#c94700) (formerly TU Kaiserslautern-Landau) and used as example in their [Saturated Min-Sum decoding](https://www.date-conference.com/proceedings-archive/2016/pdf/0760.pdf) DATE 2016 paper. 

Configuration file to reproduce the rECCT results: [here](https://github.com/lebidan/sbnd/blob/main/conf/exp/recct-ldpc-tukl-96-48-on-demand-3dB.yaml)

</details>

## 🎯 Features

* **Multiple decoder architectures** — ships with `StackedGRU`, `ECCT`, `CrossMPT`, and `rECCT` (a recurrent ECCT), all sharing a common interface
* **Easy to extend** — add your own architecture by subclassing the [shared `BaseDecoder`](https://github.com/lebidan/sbnd/blob/main/src/decoder.py) and using the [template decoder](https://github.com/lebidan/sbnd/blob/main/src/mocked.py) as a starting point
* **Two decoding modes** — standard codeword-level SBND, or message-level [iSBND](https://arxiv.org/abs/2402.13948) for non-systematic codes
* **Hydra configuration** — every aspect of training is configurable via composable YAML files
* **Flexible data pipeline** — train on pre-computed datasets or generate noisy codewords on-the-fly
* **Data augmentation** — leverage code automorphisms to increase training diversity
* **Multi-GPU** — distributed training via PyTorch Lightning DDP
* **Monte Carlo evaluation** — evaluate trained models over configurable Eb/N0 ranges with BER/WER reporting
* **Test-time scaling** — boost decoding performance at inference with sequential (self-boosting) or parallel (test-time augmentation) variants
* **Experiment tracking** — built-in CSV and [Weights & Biases](https://wandb.ai) logging with gradient/weight monitoring

## 💻 Installation

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

## 🚀 Getting Started 

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

See the [Configuration Guide](#-configuration-guide) for details on how to create your own experiments.

Training artifacts (Hydra config, logs, checkpoints) are saved under `./log/train/runs/YYYY-MM-DD_HH-MM-SS/` to make sure each run is unique. 

Two checkpoints are saved in the `checkpoints/` run subdirectory: `last.ckpt` (model state at the latest epoch) and `<exp-file-name>-<max_epochs>epochs-<wandb-run-name>.ckpt` (best model by validation accuracy). The W&B run name suffix is omitted if W&B is not installed.

### Evaluate a model

`sbnd-test` evaluates a trained checkpoint through Monte Carlo simulation over a range of Eb/N0 values, reporting **Word Error Rate (WER)** and **Bit Error Rate (BER)** (calculated on the message bits) for each SNR point. The mode (`error_space`) used at training time is read back from the checkpoint, so FER/BER are computed accordingly — see [Decoding modes](#decoding-modes).

Like `sbnd-train`, evaluation is configured via [Hydra](https://hydra.cc/). The base config [`conf/test.yaml`](https://github.com/lebidan/sbnd/blob/main/conf/test.yaml) ships with preset Monte-Carlo simulation defaults, so a first evaluation pass only needs the model checkpoint:

```
sbnd-test model=/path/to/my-model.ckpt
```

where `/path/to/my-model.ckpt` should have the form `./log/train/runs/YYYY-MM-DD_HH-MM-SS/checkpoints/<exp-file-name>-<max_epochs>epochs-<wandb-run-name>.ckpt`

Any field can be overridden directly on the command line:

```
sbnd-test model=/path/to/my-model.ckpt snr_min=1 snr_max=5 snr_step=0.5 num_batches=8192 batch_size=4096
```

For repeated evaluations with the same set of options, group them into a preset under [`conf/eval/`](https://github.com/lebidan/sbnd/tree/main/conf/eval) and select it with `eval=<name>` (e.g. one preset per code):

```
sbnd-test model=/path/to/my-model.ckpt eval=my-eval-config
```

> A few evaluation presets for the different codes shipped with SBND are available in [`conf/eval/`](https://github.com/lebidan/sbnd/tree/main/conf/eval). You may to adjust the batch size and number of batches to match your GPU capabilities.

Results are saved to a CSV file named after the checkpoint under the output directory (default: [`./log/test/`](https://github.com/lebidan/sbnd/tree/main/log/test)). If the file already exists, new SNR points are appended; for SNR points that are already present, the new error counts are **accumulated** on top of the previous ones (and WER/BER are recomputed from the cumulative totals), so you can extend an evaluation incrementally across multiple runs and tighten the statistics over time.

| Option | Default | Description |
| --- | --- | --- |
| `model` | — (required) | Path to the model checkpoint to evaluate |
| `snr_min` / `snr_max` / `snr_step` | 0.0 / 5.0 / 1.0 | Eb/N₀ range to simulate (dB) |
| `batch_size` | 4096 | Test batch size |
| `num_batches` | 1024 | Number of batches per SNR point |
| `num_workers` | 8 | Number of workers for dataloading |
| `hdd` | `false` | Emulate hard-decision decoding (perfect correction if errors ≤ t). Requires a code with known `dmin` and a model trained in `error_space=codeword` |
| `tts` | `SingleShotDecoder` | Decoding strategy — see [Test-time scaling](#test-time-scaling) |
| `output_dir` | `./log/test` | Output directory for the results CSV |

#### Test-time scaling

Beyond the no-TTS baseline (one forward pass per sample), `sbnd-test` supports two test-time scaling (TTS) variants that trade extra inference compute for lower error rates. Both can be activated through the `tts:` block in the evaluation config and require a model trained in `error_space=codeword` (the syndrome check that drives early termination only makes sense in codeword space). The active strategy is appended to the output filename so different TTS sweeps don't overwrite one another (e.g. `<model>-sb5.csv`, `<model>-tta4.csv`). TTS combines with `hdd: true`, in which case both suffixes appear (e.g. `<model>-sb5-hdd.csv`).

**1. Self-boosting** (sequential TTS, [`SelfBoostingDecoder`](https://github.com/lebidan/sbnd/blob/main/src/tts.py)). The model iterates over its own predictions in an attempt to correct the errors left at the previous iteration. The loop stops as soon as a sample's prediction passes the syndrome check, or after a maximum of `num_iters` model invocations. Early references to such a strategy are the *Iterative Error Correction* approach of [Kavvousanos & Paliouras, GLOBECOM 2020](https://ieeexplore.ieee.org/document/9367553) and the *Iterative Error Decimation* decoder by [Kamassury & Silva (2021)](https://arxiv.org/abs/2012.00089).

```yaml
tts:
  _target_: sbnd.tts.SelfBoostingDecoder
  num_iters: 5
```

**2. Test-time augmentation** (parallel TTS, [`TTADecoder`](https://github.com/lebidan/sbnd/blob/main/src/tts.py)). For each test sample, draw `num_perms` random code automorphisms from the same `transforms.py` classes used for training-time data augmentation (`BCHPerms`, `QCPerms`, `GenericPerms`). Each permutation is applied to the received word, the model is run, and the resulting logits are inverse-permuted back to the original coordinates. If one of the predictions passes the syndrome check, this is the model output. Otherwise the output logits are calculated as the average of all predictions.

```yaml
tts:
  _target_: sbnd.tts.TTADecoder
  num_perms: 4
  transform:
    _partial_: true
    _target_: sbnd.transforms.BCHPerms   # same classes as the training transform
    is_extended: false                    # set true for eBCH codes
```

Both TTS variants can be activated either by adding the block above to a [`conf/eval/`](https://github.com/lebidan/sbnd/tree/main/conf/eval) preset or directly on the command line:

```
sbnd-test model=/path/to/my-model.ckpt eval=bch-31-21 \
  tts._target_=sbnd.tts.SelfBoostingDecoder +tts.num_iters=10
```

The self-boosting and TTA strategies have been compared in the [PhD thesis of A. Ismail, Chap. 4.2](https://theses.fr/2025IMTA0515). Both rapidly increase the inference cost and show diminishing returns as the model gets better. Whenever applicable, hard-decision decoding (`hdd` emulation flag) remains the most cost-efficient and effective strategy to get an extra boost in performance at inference time.


## 🔍 Supported Codes & Decoders

### Codes

A collection of standard BCH, extended BCH, QC-LDPC, Reed-Muller and Polar codes are shipped in [`data/codes/`](https://github.com/lebidan/sbnd/tree/main/data/codes). Any linear code can be used by providing a MATLAB `.mat` file with the following fields:

| Field | Required | Description |
| --- | --- | --- |
| `n` | ✓ | Code length |
| `k` | ✓ | Message length |
| `G` | ✓ | Generator matrix (k × n) |
| `H` | ✓ | Parity-check matrix (m × n, with m ≥ n-k)|
| `Ginv` |  | Inverse-encoding matrix (n × k) such that `G · Ginv = I_k` — used to recover the message from a decoded codeword, and required to train/evaluate in message-level mode (see [Decoding modes](#decoding-modes)). If omitted, it is built automatically when the generator matrix is systematic (identity block at the beginning or end of `G`). For **non-systematic** codes, `Ginv` must be provided explicitly in the `.mat` file. |
| `dmin` |  | Minimum distance (defaults to `None` if not provided) |
| `name` |  | Code family name (defaults to `"Linear"`) |

Reed-Muller and Polar codes are examples of codes with a non-systematic encoder and for which the code file includes a reverse-encoding matrix `Ginv`.

### Decoder architectures

SBND ships with four syndrome-based neural decoder architectures:

| Decoder | Class | Source | Reference |
| --- | --- | --- | --- |
| StackedGRU | `sbnd.gru.StackedGRU` | [`gru.py`](https://github.com/lebidan/sbnd/blob/main/src/gru.py) | [Bennatan et al., 2018](https://arxiv.org/abs/1802.04741) |
| ECCT | `sbnd.ecct.ECCT` | [`ecct.py`](https://github.com/lebidan/sbnd/blob/main/src/ecct.py) | [Choukroun & Wolf, 2022](https://arxiv.org/abs/2206.14881) |
| CrossMPT | `sbnd.crossmpt.CrossMPT` | [`crossmpt.py`](https://github.com/lebidan/sbnd/blob/main/src/crossmpt.py) | [Park et al., 2025](https://arxiv.org/abs/2507.01038) |
| rECCT | `sbnd.recct.RECCT` | [`recct.py`](https://github.com/lebidan/sbnd/blob/main/src/recct.py) | [de Boni Rovella, 2024](https://theses.fr/2024ESAE0065) |

<details><summary>GRU decoder</summary>

The stacked GRU decoder is the straightforward implementation of [Bennatan et al.'s (2018) architecture](https://arxiv.org/abs/1802.04741). The syndrome and LLR magnitude vectors are concatenated to form an input vector that is repeated at each time step, unless parameter `zero_padding=True`. In the latter case, the input vector is fed only at the first time step and an all-zero input vector is used at all subsequent time steps. The error pattern estimate is obtained by passing the output of the last time step through a linear layer. We have found that better performance is obtained by using very few layers (2) and more time steps rather than the deeper 5-layer architecture used in the original paper.

</details>

<details><summary>ECCT decoder</summary>

Essentially the verbatim copy of the implementation published in the [original repo](https://github.com/yoniLc/ECCT). The two main changes are the use of PyTorch's `scaled_dot_product_attention` function to speed up training, and a mask modified to prevent tokens to attend to themselves. The latter was found to slightly improve accuracy in our experiments.

</details>

<details><summary>CrossMPT decoder</summary>

Verbatim copy of the implementation published in the [original repo](https://github.com/iil-postech/crossmpt), with the use of PyTorch's `scaled_dot_product_attention` function to speed up training.

</details>

<details><summary>rECCT decoder</summary>

The rECCT decoder is a recurrent implementation of ECCT which can reach comparable performance with fewer parameters (up to 10x less in certain cases). It was inspired by the [PhD work](https://theses.fr/2024ESAE0065) of Gaston de Boni Rovella. There has been renewed interest recently in recurrent transformers as a parameter-efficient architecture (see, e.g., the many papers on looped transformers that have flourished on arXiv since 2025).

</details>

All decoders inherit from the abstract [`BaseDecoder`](https://github.com/lebidan/sbnd/blob/main/src/decoder.py) class in [`src/decoder.py`](https://github.com/lebidan/sbnd/blob/main/src/decoder.py), which defines the shared interface: `forward(ym, s) → logits`, where `ym` is the normalized channel magnitude `|y|/max(|y|)`, `s` is the bipolar syndrome vector, and `logits` is the decoder prediction of the target error pattern. `BaseDecoder` also centralizes the common constructor arguments (`code`, `error_space`, `compile`) and the standard attributes (`output_sz`, `example_input_array`) — see the header of [`src/decoder.py`](https://github.com/lebidan/sbnd/blob/main/src/decoder.py) for the full API description, including the meaning of `error_space` and the convention of calling `self._maybe_compile()` last in the subclass `__init__`.

To implement your own decoder, inherit from `BaseDecoder` and use [`src/mocked.py`](https://github.com/lebidan/sbnd/blob/main/src/mocked.py) as a minimal starting template.

### Decoding modes

SBND supports two decoding modes, selected via the shared `error_space` parameter on both the decoder and the datamodule (they must agree, and a mismatch is caught at `trainer.fit` start):

1. **Codeword-level decoding** (`error_space: "codeword"`, default) — the standard SBND setup. The decoder is trained to predict the full n-bit error pattern `e_cw = c - c_hat` affecting the transmitted codeword. Evaluation reports the **FER on the decoded codeword** and the **BER on the decoded message**, with the codeword-to-message mapping inverted via `Ginv` when the code is non-systematic.

2. **Message-level decoding** (`error_space: "message"`), which we abbreviate as **iSBND** (information-based SBND), proposed in [De Boni Rovella & Benammar, GLOBECOM 2023](https://arxiv.org/abs/2402.13948). The decoder directly estimates the k-bit error pattern on the information message, computed as `e_msg = Ginv · e_cw`. This mode is particularly well-suited to non-systematic codes, but can also bring a small FER gain on systematic codes: it is generally slightly easier for the model to learn the partial error pattern restricted to the first or last k bits of the codeword (the message part), than the full n-bit error pattern. For models trained in iSBND mode, evaluation reports both **FER and BER on the decoded message**.

Both the decoder and the datamodule default to `"codeword"`, so standard SBND experiments need no extra config. To switch to iSBND mode, set `error_space: "message"` on both the `decoder:` and `data:` blocks of your experiment config.

## 📝 Configuration Guide 

Training is orchestrated by [`SBNDLitModule`](https://github.com/lebidan/sbnd/blob/main/src/model.py), a PyTorch Lightning module wrapper. The SBND decoder architecture to train is passed as a constructor argument to this module. SBND models are trained in a supervised manner, to minimize the average binary cross-entropy between the predicted and target error patterns. The two main metrics monitored during training are **loss** and **accuracy** (the fraction of correctly predicted error patterns).

Training configuration is managed with [Hydra](https://hydra.cc). The base config [`conf/train.yaml`](https://github.com/lebidan/sbnd/blob/main/conf/train.yaml) defines defaults for hardware, logging, callbacks, and path variables such as `codes_dir` (default: `./data/codes`). Experiment configs under [`conf/exp/`](https://github.com/lebidan/sbnd/tree/main/conf/exp) override what they need, following the naming convention `<decoder>-<code>-<data_mode>-<dataset_size>-<snr>[-aug].yaml`. The `dev-test-mocked` experiment is an exception to this convention: it serves as a quick sanity check and is the default when no experiment is specified.

> We recommend starting from the shipped examples in [`conf/exp/`](https://github.com/lebidan/sbnd/tree/main/conf/exp) and adapting them to your needs. Each training experiment comes with a model performance evaluation log file in [`log/test`](https://github.com/lebidan/sbnd/tree/main/log/test).

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

At present, SBND supports two training data strategies, selected by whether `train_file` is set:

#### 1. On-demand generation (default, no `train_file`)

Noisy codewords are generated randomly at every training step. The model is exposed to fresh data at each step — no sample is ever repeated. This mode avoids large dataset files and is the simplest way to get started. Note that data augmentation is not applied in this mode, since the data is already unique at every step. The downside is that the model is trained for perfect correction, an unrealistic goal that ultimately hinders WER performance (see our [ICMLCN 2025 paper](https://arxiv.org/abs/2502.10183) for details). In this mode, a training epoch consists of `n_train_samples / train_bs` training steps, or batches.

On-demand mode is selected automatically when `train_file` is omitted; it requires `ebno_dB_train` and a non-zero `n_train_samples`. If `n_val_samples` is not given, validation defaults to 25% of `n_train_samples`. Both `n_train_samples` and `n_val_samples` are rounded down to the nearest multiple of `train_bs` / `val_bs`.

```yaml
data:
  _target_: sbnd.data.SBNDDataModule
  ebno_dB_train: 2.0
  n_train_samples: 1048576
  train_bs: 4096
  n_val_samples: 524288
  val_bs: 4096
```

#### 2. Pre-computed datasets (`train_file` specified)

Load training and validation data from user-supplied `.mat` files. Each file must contain a matrix of received words `y` and a matrix of target binary error patterns `e`. The same fixed dataset is reused at each epoch, which gives total control over the training distribution, making it possible to more closely approach Maximum Likelihood decoding performance with [much fewer samples than with on-demand data](https://arxiv.org/abs/2502.10183). If no `val_file` is provided, a validation set is created by random split of the training set — the default split ratio is 75%/25%, overridable with an explicit `n_val_samples`. The training transform (if any) is applied only to the training subset; validation samples are never augmented.

`n_train_samples` defaults to 0, meaning the entire file is used; set it to a positive value to use only the first N rows. `n_val_samples` behaves identically when loading from `val_file`.

The `data_dir` variable defaults to `./data/datasets` and is defined in [`conf/train.yaml`](https://github.com/lebidan/sbnd/blob/main/conf/train.yaml).

```yaml
data:
  _target_: sbnd.data.SBNDDataModule
  train_file: ${data_dir}/bch-63-45/train-ml-4M-2dB.mat
  train_bs: 4096
  val_file: ${data_dir}/bch-63-45/val-ml-512K-2dB.mat
  val_bs: 4096
```

**Forbidden combinations.** Setting `val_file` without `train_file` is rejected at construction time (on-demand mode does not load validation from a file). On-demand mode silently ignores `transform`, since each batch is already unique.

#### Pre-computed dataset format and download

Pre-computed training datasets are too large to ship with the repository. Most of the training experiments in the [`conf/exp/`](https://github.com/lebidan/sbnd/tree/main/conf/exp) directory can be reproduced with the datasets listed below. Each dataset consists of ML error patterns collected by standard Monte Carlo simulation of an ordered statistics decoder (OSD), and comes as a bundle of training and validation data.

| Code | Dataset description | Size | Link |
| --- | --- | --- | --- |
| RM(32,16,8) | 4M training + 512K validation samples collected at Eb/N0 = 3 dB | ~1 GB | [Download](https://sdrive.cnrs.fr/s/waykmQteWx5RZPn) |
| eBCH(32,16,8) | 4M training + 512K validation samples collected at Eb/N0 = 3 dB | ~1 GB | [Download](https://sdrive.cnrs.fr/s/fx7kN9s5MwZfi35) |
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

For pre-computed datasets (mode 2 above), data augmentation can be enabled via the `transform` option. This applies random permutations from the code's automorphism group to each batch, effectively multiplying the number of distinct training examples. At present, the following code-specific transforms are available in [`transforms.py`](https://github.com/lebidan/sbnd/blob/main/src/transforms.py):

* [`BCHPerms`](https://github.com/lebidan/sbnd/blob/main/src/transforms.py) — cyclic × Frobenius permutations for BCH codes (works with extended BCH too by setting `is_extended = true`)
* [`QCPerms`](https://github.com/lebidan/sbnd/blob/main/src/transforms.py) — quasi-cyclic shift permutations for QC-LDPC codes (requires the circulant size `Zc`)

```yaml
data:
  transform:
    _partial_: true
    _target_: sbnd.transforms.BCHPerms   # or sbnd.transforms.QCPerms
    is_extended: true # for eBCH codes only
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

### Test evaluation

`sbnd-train` automatically runs a final test evaluation on the best checkpoint at the end of every training run (`trainer.test` is called after `trainer.fit`). Test data is always generated on-the-fly at each Eb/N0 value listed in `ebno_dB_test`, and the following metrics are reported and logged for each SNR point:

- `test/loss` — cross-entropy loss on the test set
- `test/acc` — fraction of correctly predicted error patterns (word accuracy)
- `test/err` — word error rate (WER = 1 − acc)

The test SNR range, sample count, and batch size are configured under the `data:` block of the experiment config:

```yaml
data:
  ebno_dB_test: [2.0, 3.0, 4.0]   # one test set per SNR value
  n_test_samples: 2097152           # 2M samples per SNR point (default)
  test_bs: 4096
```

`n_test_samples` is rounded down to the nearest multiple of `test_bs`. In addition, the `PeriodicTest` callback runs a lightweight interim test evaluation every `every_n_epochs` epochs (default: 50) during training, logging results under the `periodic_test/` namespace. This allows monitoring test-set progress without waiting for the full training run to complete. The interval can be changed in the experiment config:

```yaml
periodic_test_cb:
  _target_: sbnd.train.PeriodicTest
  every_n_epochs: 100   # or 0 to disable
```

> For more comprehensive model evaluation including bit-error rate performance and access to different test-time scaling strategies, use the `sbnd-test` command as described in the [Getting Started](#-getting-started) section.

## 📁 Project Structure

```
sbnd/
├── conf/
│   ├── train.yaml              # Base Hydra config for training (hardware, logging, callbacks, path variables)
│   ├── test.yaml               # Base Hydra config for evaluation (Monte-Carlo simulation defaults)
│   ├── exp/                    # Training experiment configs
│   └── eval/                   # Evaluation presets
├── data/
│   └── codes/                  # Code definition .mat files (G, H, n, k)
├── media/                      # Logo, plots, etc.
├── src/                        # Python package (installed as `sbnd`)
│   ├── codes.py                # LinearCode class
│   ├── data.py                 # SBNDDataModule: datasets and batch generation
│   ├── model.py                # SBNDLitModule: Lightning training wrapper
│   ├── decoder.py              # BaseDecoder: shared decoder API
│   ├── ecct.py                 # ECCT decoder
│   ├── crossmpt.py             # CrossMPT decoder
│   ├── recct.py                # rECCT decoder
│   ├── gru.py                  # StackedGRU decoder
│   ├── mocked.py               # Minimal template decoder
│   ├── transforms.py           # Code automorphisms (BCHPerms, QCPerms, GenericPerms) — used for both training-time augmentation and TTA
│   ├── tts.py                  # Decoding strategies: no-TTS baseline, SelfBoosting, TTA
│   ├── lr_sched.py             # LR schedulers (CosineWarmupLR, WarmupStableDecayLR)
│   ├── train.py                # sbnd-train entry point
│   ├── test.py                 # sbnd-test entry point
│   └── utils.py                # Logging utilities
├── pyproject.toml              # Package metadata and dependencies
└── LICENSE                     # MIT License
```

## ⚖️ License

This project is licensed under the [MIT License](https://github.com/lebidan/sbnd/blob/main/LICENSE).

## 🛠️ Contributing

Contributions are welcome. Please open an [issue](https://github.com/lebidan/sbnd/issues) to report bugs or suggest features, and feel free to submit pull requests.

## 🤝 Acknowledgments

Much of this code was developed within the framework of the [ANR-21 AI4CODE project](https://ai4code.projects.labsticc.fr/).

The following decoder implementations are adapted from their original authors' code:

* **ECCT** — adapted from [yoniLc/ECCT](https://github.com/yoniLc/ECCT) (MIT License), by Y. Choukroun and L. Wolf
* **CrossMPT** — adapted from [iil-postech/crossmpt](https://github.com/iil-postech/crossmpt), by S.-J. Park et al.

You may want also to pay a visit to [Gaston de Boni Rovella's github repository](https://github.com/gastondeboni/Syndrome_Based_Neural_Decoding) for another SBND decoding implementation based on [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/).

This project has greatly benefited from the following open-source software:

* [PyTorch](https://pytorch.org/) — deep learning framework
* [Lightning](https://lightning.ai/) — training infrastructure and multi-GPU support
* [Hydra](https://hydra.cc/) — configuration management

## Citation

If you find this code helpful in your project or research, please consider citing it:

```bibtex
@misc{lebidan2026sbnd,
      title={SBND: Syndrome-based neural decoding of linear error-correcting codes}, 
      author={Raphaël Le Bidan},
      year={2026},
      url={https://github.com/lebidan/sbnd}, 
}
```