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
<a href="#-documentation">Documentation</a> |
<a href="docs/training.md">Training</a> |
<a href="docs/evaluation.md">Evaluation</a> |
<a href="docs/extending.md">Extending</a> |
<a href="#-contributing">Contributing</a>
</p>

**`SBND`** is a PyTorch/Lightning framework for training and evaluating syndrome-based neural decoders for linear error-correcting codes.

---

## 👀 Why SBND? 

Syndrome-based neural decoding is a promising approach for soft-decision decoding of short, high-rate codes, but the field is still wide open. Performance lags behind classical decoders like OSD or Chase-2, scaling laws are poorly understood, and more parameter-efficient architectures are yet to be found.

`SBND` is built for researchers who want to close that gap. It ships with multiple architectures, reproducible baselines, and a clean training infrastructure — everything you need to run experiments, test new ideas, and push neural decoders further than they've been before. 

<b> ⭐ Performance highlights ⭐</b>

<details><summary>Decoding the (63,45,7) BCH code</summary>

<img alt="BCH(63,45,7) performance" src="https://raw.githubusercontent.com/lebidan/sbnd/main/media/fer_63_45.png?raw=true" width=90%>

- Training the original ECCT with SBND brings ~two-order of magnitude WER improvement
- Same or better performance with half the number of parameters when switching to our recurrent ECCT model
- Performance is within 0.2 dB of MLD and matches Chase-2 decoding with 64 test patterns

Configuration files for the above experiments: [original/improved ECCT training](https://github.com/lebidan/sbnd/blob/main/conf/exp/ecct-bch-63-45-on-demand-2dB.yaml), [rECCT training](https://github.com/lebidan/sbnd/blob/main/conf/exp/recct-bch-63-45-ml-4m-2dB-aug.yaml)

</details>

<details><summary>Decoding the (32,16,8) extended BCH code</summary>

<img alt="eBCH(32,16,8) performance" src="https://raw.githubusercontent.com/lebidan/sbnd/main/media/fer_32_16.png?raw=true" width=90%>

- WER performance within 0.2 dB of MLD and comparable to Chase-2 decoding with 64 test patterns
- Outperforms the original ECCT and CrossMPT decoders with 8x fewer parameters

Note: The comparison between results for the (31,16,7) and (32,16,8) codes is reasonable as both codes have very close MLD performance down to WER = 1E-4. The extended code progressively takes over at high SNRs. Compare with the results in Table 3 and Fig. 11 from [the CrossMPT ICLR 2025 paper](https://openreview.net/forum?id=gFvRRCnQvX).

Configuration file to reproduce the rECCT results: [here](https://github.com/lebidan/sbnd/blob/main/conf/exp/recct-ebch-32-16-ml-16m-3dB.yaml)

</details>

<details><summary>Decoding the (96,48,10) quasi-cyclic LDPC code</summary>

<img alt="QC-LDPC(96,48,10) performance" src="https://raw.githubusercontent.com/lebidan/sbnd/main/media/fer_96_48.png?raw=true" width=90%>

- High-SNR WER performance within 1.0 dB or less of MLD (but still *much room left for improvement*)
- Matches or outperforms BP with 100 iterations

This very nice and strong short quasi-cyclic LDPC code was designed at [RPTU](https://rptu.de/channel-codes/channel-codes-database/more-ldpc-codes#c94700) and used as example in their [Saturated Min-Sum decoding](https://www.date-conference.com/proceedings-archive/2016/pdf/0760.pdf) DATE 2016 paper. 

Configuration file to reproduce the rECCT results: [here](https://github.com/lebidan/sbnd/blob/main/conf/exp/recct-ldpc-rptu-96-48-on-demand-3dB.yaml)

</details>

## 🎯 Features

* **Multiple decoder architectures** — ships with `StackedGRU`, `ECCT`, `CrossMPT`, and `rECCT` (a recurrent ECCT), all sharing a common interface
* **Easy to extend** — add your own architecture by subclassing the [shared `BaseDecoder`](https://github.com/lebidan/sbnd/blob/main/src/decoder.py) and using the [template decoder](https://github.com/lebidan/sbnd/blob/main/src/mocked.py) as a starting point
* **Two decoding modes** — standard codeword-level SBND, or message-level [iSBND](https://arxiv.org/abs/2402.13948) for non-systematic codes
* **Hydra configuration** — every aspect of training is configurable via composable YAML files
* **Flexible data pipeline** — train on pre-computed datasets, generate noisy codewords on the fly (with optional multi-SNR sampling), or mix both within each batch
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

Training is configured with [Hydra](https://hydra.cc). Each experiment config under [`conf/exp/`](conf/exp) defines a complete training setup: the error-correcting **code**, the **decoder** architecture, the **training data** pipeline, and the **training parameters** (optimizer, LR scheduler, precision, etc.). Launch a training job with the `sbnd-train` CLI, selecting an experiment with `exp=`:

```
sbnd-train exp=ecct-bch-63-45-ml-4m-2dB-aug
```

Any config value can be overridden on the command line:

```
sbnd-train exp=ecct-bch-63-45-ml-4m-2dB-aug gpus=2 cpus=16 max_epochs=64 lr=0.001
```

Training artifacts (Hydra config, logs, checkpoints) are saved under `./log/train/runs/YYYY-MM-DD_HH-MM-SS/`. Two checkpoints are written to the `checkpoints/` run subdirectory: `last.ckpt` (latest epoch) and `<exp-file-name>-<max_epochs>epochs-<wandb-run-name>.ckpt` (best model by validation accuracy).

→ Full reference, including how to create your own training experiment config: see [docs/training.md](docs/training.md).

### Evaluate a model

`sbnd-test` evaluates a trained checkpoint through Monte-Carlo simulation over a range of Eb/N0 values, reporting **Word Error Rate (WER)** and **Bit Error Rate (BER)** at each SNR point. A first evaluation pass requires only the model checkpoint:

```
sbnd-test model=/path/to/my-model.ckpt
```

Repeated evaluations with the same set of options can be grouped into a preset under [`conf/eval/`](conf/eval) and selected with `eval=<name>`:

```
sbnd-test model=/path/to/my-model.ckpt eval=bch-31-21
```

`sbnd-test` also supports hard-decision decoding emulation (the `hdd` flag) and two test-time scaling variants — sequential **self-boosting** and parallel **test-time augmentation** — that exchange extra inference compute for lower error rates.

→ Full reference, options table, and TTS configuration: see [docs/evaluation.md](docs/evaluation.md).


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

An almost verbatim port of the official ECCT implementation published in the [original repo](https://github.com/yoniLc/ECCT). The two main changes are the use of PyTorch's [`scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) function to speed up training, and a mask modified to prevent tokens to attend to themselves. The latter was found to slightly improve accuracy in our experiments.

</details>

<details><summary>CrossMPT decoder</summary>

An almost verbatim port of the official CrossMPT implementation published in the [original repo](https://github.com/iil-postech/crossmpt), with the use of PyTorch's [`scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) function to speed up training.

</details>

<details><summary>rECCT decoder</summary>

The rECCT decoder is a recurrent implementation of ECCT which can reach comparable performance with fewer parameters (up to 10x less in certain cases). It was inspired by the [PhD work](https://theses.fr/2024ESAE0065) of Gaston de Boni Rovella. There has been renewed interest recently in recurrent transformers as a parameter-efficient architecture (see, e.g., the many papers on looped transformers that have flourished on arXiv since 2025).

</details>

All decoders inherit from the abstract [`BaseDecoder`](https://github.com/lebidan/sbnd/blob/main/src/decoder.py) class in [`src/decoder.py`](https://github.com/lebidan/sbnd/blob/main/src/decoder.py), which defines the shared interface: `forward(ym, s) → logits`, where `ym` is the normalized channel magnitude `|y|/max(|y|)`, `s` is the bipolar syndrome vector, and `logits` is the decoder prediction of the target error pattern. `BaseDecoder` also centralizes the common constructor arguments (`code`, `error_space`, `compile`) and the standard attributes (`output_sz`, `example_input_array`) — see the header of [`src/decoder.py`](https://github.com/lebidan/sbnd/blob/main/src/decoder.py) for the full API description, including the meaning of `error_space` and the convention of calling `self._maybe_compile()` last in the subclass `__init__`.

To implement your own decoder, inherit from `BaseDecoder` and use [`src/mocked.py`](src/mocked.py) as a minimal starting template — see [docs/extending.md](docs/extending.md) for the full description of the interface and conventions.

### Decoding modes

SBND supports two decoding modes, selected via the shared `error_space` parameter on both the decoder and the datamodule (they must agree, and a mismatch is caught at `trainer.fit` start):

1. **Codeword-level decoding** (`error_space: "codeword"`, default) — the standard SBND setup. The decoder is trained to predict the full n-bit error pattern `e_cw = c - c_hat` where `c` is the transmitted codeword and `c_hat` is the decoder decision on `c`.. Evaluation reports the **WER on the decoded codeword** and the **BER on the decoded message**, with the codeword-to-message mapping inverted via `Ginv` when the code is non-systematic.

2. **Message-level decoding** (`error_space: "message"`), which we abbreviate as **iSBND** (information-based SBND), proposed in [De Boni Rovella & Benammar, GLOBECOM 2023](https://arxiv.org/abs/2402.13948). The decoder directly estimates the k-bit error pattern on the information message, computed as `e_msg = Ginv · e_cw`. This mode is particularly well-suited to non-systematic codes, but can also bring a small WER gain on systematic codes: it is generally slightly easier for the model to learn the partial error pattern restricted to the first or last k bits of the codeword (the message part), than the full n-bit error pattern. For models trained in iSBND mode, evaluation reports both **WER and BER on the decoded message**.

Both the decoder and the datamodule default to `"codeword"`, so standard SBND experiments need no extra config. To switch to iSBND mode, set `error_space: "message"` on both the `decoder:` and `data:` blocks of your experiment config. See for example this [Reed-Muller decoding experiment](https://github.com/lebidan/sbnd/blob/main/conf/exp/recct-rm-32-16-ml-4m-3dB.yaml).

## 📚 Documentation

The reference documentation is split into three focused guides under [`docs/`](docs):

* [**Training a model**](docs/training.md) — creating a training experiment config: specifying code, data (on-demand, pre-computed, or a mix of both; multi-SNR sampling; augmentation; dataset format and download), decoder, optimizer/scheduler, precision, resume vs. continue, logging, and end-of-training test evaluation.
* [**Evaluating a model**](docs/evaluation.md) — running `sbnd-test`: the basic Monte-Carlo SNR sweep to measure WER and BER, hard-decision decoding optional post-filtering, and the test-time scaling variants (self-boosting and TTA).
* [**Extending SBND**](docs/extending.md) — adding your own decoder architecture: the `BaseDecoder` template, conventions, a walk-through of the mocked decoder example, and how to wire it into an experiment.

## 📁 Project Structure

```
sbnd/
├── conf/                       # Hydra configs — base train.yaml/test.yaml + exp/ presets and eval/ presets
├── data/
│   ├── codes/                  # Code definition .mat files (G, H, n, k)
│   └── perms/                  # Code automorphism .mat files (used by sbnd.transforms.GenericPerms)
├── docs/                       # Reference guides: training.md, evaluation.md, extending.md
├── media/                      # Logo, plots, etc.
├── src/                        # Python package (installed as `sbnd`)
│   ├── codes.py                # LinearCode class
│   ├── data.py                 # SBNDDataModule: datasets and batch generation
│   ├── model.py                # SBNDLitModule: Lightning training wrapper
│   ├── decoder.py              # BaseDecoder: shared decoder API
│   ├── ecct.py / crossmpt.py / recct.py / gru.py / mocked.py   # Decoder implementations
│   ├── transforms.py           # Code automorphisms (BCHPerms, QCPerms, GenericPerms)
│   ├── tts.py                  # Decoding strategies: no-TTS baseline, SelfBoosting, TTA
│   ├── lr_sched.py             # LR schedulers (CosineWarmupLR, WarmupStableDecayLR)
│   ├── train.py / test.py      # sbnd-train and sbnd-test entry points
│   └── utils.py                # Logging utilities
├── pyproject.toml              # Package metadata and dependencies
├── CITATION.cff                # Citation metadata (used by GitHub's "Cite this repository")
└── LICENSE                     # MIT License
```

## ⚖️ License

This project is licensed under the [MIT License](https://github.com/lebidan/sbnd/blob/main/LICENSE).

## 🛠 Contributing

Contributions are welcome. Please open an [issue](https://github.com/lebidan/sbnd/issues) to report bugs, suggest features (new codes, new decoders, etc), or propose better results or training parameters for the available codes and decoders. We'd be happy to update this codebase according to your feedback. 

## 🤝 Acknowledgments

Much of this code was developed within the framework of the [ANR-21 AI4CODE project](https://ai4code.projects.labsticc.fr/). Earlier versions of this codebase were used to obtain the results reported in our [ICMLCN 2025 paper](https://arxiv.org/abs/2502.10183) as well as some of the results presented in Chapters 3 and 4 of [Ahmad Ismail PhD thesis](https://theses.fr/2025IMTA0515).

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