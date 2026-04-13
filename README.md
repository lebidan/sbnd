# SBND — Syndrome-Based Neural Decoding

A PyTorch/Lightning framework for training and evaluating syndrome-based neural decoders for linear error-correcting codes. 

Documentation in early stage: to be updated very soon. Stay tuned. 

## Key features

- Ship with several decoder architectures: `StackedGRU`, `ECCT`, `CrossMPT`, `rECCT`
- Easy to extend with your own architectures
- Highly configurable training and evaluation functions
- Train on fixed datasets or with on-demand data
- Data augmentation through code automorphisms

## Installation

Clone the repository first — `conf/` (Hydra experiment configs) is not distributed
in the wheel, so the commands below must be run from the repo root.

### uv pip (recommended)

`uv pip install` supports `--torch-backend=auto`, which queries the local 
accelerator available and fetches the matching PyTorch wheel automatically, 
including CPU/MPS on Apple Silicon. Replace `".[wandb]"` with `.` if you do not
want to use [Weights & Biases](https://wandb.ai).

```bash
git clone https://github.com/lebidan/sbnd.git
cd sbnd
uv venv                                              # create .venv
source .venv/bin/activate                            # on Windows: .venv\Scripts\activate
uv pip install -e ".[wandb]" --torch-backend=auto   # CUDA auto-detected, incl. W&B
```

You can also pin a specific CUDA version explicitly:

```bash
uv pip install -e ".[wandb]" --torch-backend=cu128   # force CUDA 12.8 wheel
```

To install also the development tools `black` and `mypy`:

```bash
uv pip install black mypy
```

### plain pip

Since PyTorch 2.6, CUDA-capable wheels are published directly to PyPI, so no
extra index is required on Linux / Windows:

```bash
git clone https://github.com/lebidan/sbnd.git
cd sbnd
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -e ".[wandb]"   # CUDA wheel served by PyPI on Linux/Windows; CPU/MPS on macOS
```

To pin a specific CUDA variant (e.g. when the PyPI default does not match your
driver), pass `--extra-index-url` explicitly:

```bash
pip install -e ".[wandb]" --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e ".[wandb]" --extra-index-url https://download.pytorch.org/whl/cu131
```

To install the development tools:

```bash
pip install black mypy
```

### Running the linter and type checker

```bash
black src/           # auto-format
black --check src/   # check only (CI)
mypy src/            # type checking
```

## Usage

### Train

Training is configured with [Hydra](https://hydra.cc). Select an experiment with `exp=`:

```bash
sbnd-train exp=ecct-bch-63-45-ml-4m-2dB-aug
```

Any config value can be overridden on the command line:

```bash
sbnd-train exp=ecct-bch-63-45-ml-4m-2dB-aug gpus=2 max_epochs=64 lr=0.001
```

### Test

Evaluate a trained checkpoint over a range of Eb/N0 values:

```bash
sbnd-test <checkpoint> --snr_min 1 --snr_max 5 --snr_step 0.5 --output results.csv
```

Key options:

| Option | Description |
|---|---|
| `--code` | Override the code `.mat` file path stored in the checkpoint |
| `--snr_min/max/step` | Eb/N0 range to simulate (dB) |
| `--batch_size` | Test batch size (default: 4096) |
| `--num_batches` | Batches per SNR point (default: 1024) |
| `--output` | Path to output CSV file |

## Data

Training data can be provided in two modes, controlled by `on_demand` in the experiment config:

**Pre-computed** (`on_demand: false`): Load pre-generated `.mat` files for training and validation.
Faster I/O at the cost of disk space. Well-suited for long training runs with a fixed SNR point.

**On-demand** (`on_demand: true`): Generate noisy codewords on-the-fly during training.
Avoids large dataset files and allows the training SNR to vary dynamically; uses more CPU.

### Data Augmentation

For codes with cyclic or quasi-cyclic symmetry, the `transform` option applies random permutations to each batch, effectively multiplying the number of distinct training examples:

- `BCHPerms` — cyclic permutations for BCH codes
- `QCPerms` — quasi-cyclic permutations for QC-LDPC codes

## Configuration

Configuration is managed with [Hydra](https://hydra.cc).
The base config `conf/train.yaml` sets defaults for hardware, logging, and callbacks.
Experiment configs under `conf/exp/` override what they need.

Key top-level parameters:

| Parameter | Description |
|---|---|
| `nodes` / `gpus` / `cpus` | Distributed training hardware |
| `max_epochs` / `lr` | Training schedule |
| `seed` | Global reproducibility seed (default: 1234) |
| `offline` | Run WandB in offline mode |
| `project` | WandB project name |

Output for each run is written to `log/train/runs/<timestamp>/`, containing checkpoints, a CSV metrics file, and a Hydra config snapshot.

## Monitoring

Each training run logs to two backends:

- **CSV** — `metrics.csv` in the run's output directory
- **Weights & Biases** — set `WANDB_API_KEY` and use `offline=false`; use `offline=true` for air-gapped environments

Monitored quantities include training/validation loss and accuracy, gradient and weight norms per layer, and Adam effective step sizes.

Checkpoints are saved under `log/train/runs/<timestamp>/checkpoints/`, keyed by best `val/acc`.
