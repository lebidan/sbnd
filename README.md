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

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/lebidan/sbnd.git
cd sbnd

uv sync               # runtime dependencies
uv sync --group dev   # + black and mypy for development
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
