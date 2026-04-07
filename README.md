# SBND — Syndrome-Based Neural Decoding

A PyTorch/Lightning framework for training and evaluating syndrome-based neural decoders for linear error-correcting codes. 

## Background

In digital communications, a linear block code encodes k information bits into an n-bit codeword using a generator matrix G.
The receiver observes a noisy version y = c + e through an AWGN channel.
Classical decoding computes the **syndrome** `s = H · ŷ ∈ {0,1}^m` — a binary vector that is zero for a valid codeword and otherwise captures information about the error pattern.

First introduced in [this paper](https://arxiv.org/abs/1802.04741), SBND decoders are deep learning models trained to infer the error pattern e from the noisy received word y. Specifically, SBND decoders take as input the received symbol magnitudes `|y|` and the binary syndrome `s`, and output LLR estimates `LLR(e)` for the error pattern.

## Key features

- Two decoder architectures: `StackedGRU` and `ECCT`
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

Available experiments:

| Experiment | Description |
|---|---|
| `ecct-bch-63-45-ml-4m-2dB-aug` | ECCT on BCH(63,45), pre-computed training data, BCH permutation augmentation |
| `ecct-bch-63-45-true-4m-2dB-aug` | ECCT on BCH(63,45), on-demand training data, BCH permutation augmentation |
| `dev-test-ecct` | Small ECCT config for quick iteration |
| `dev-test-gru` | Small GRU config for quick iteration |

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



## Decoders

| Model | Description |
|---|---|
| `ECCT` | Transformer encoder with code-aware attention masking, adapted from [ECCT](https://github.com/yoniLc/ECCT) |
| `StackedGRU` | Stacked GRU decoder unrolled over multiple decoding steps |

### ECCT

A transformer encoder where each token corresponds to a code bit.
An attention mask derived from the parity-check matrix H restricts each bit's attention to its connected check nodes, giving the model an inductive bias aligned with the Tanner graph of the code.

Key hyperparameters:

| Parameter | Description |
|---|---|
| `n_layers` | Number of transformer layers |
| `embed_dim` | Token embedding dimension |
| `n_heads` | Number of attention heads |
| `res_dropout` / `attn_dropout` / `ffn_dropout` | Dropout rates |
| `use_fast_attn` | Use `torch.nn.functional.scaled_dot_product_attention` |
| `compile` | Compile with `torch.compile` for faster training |

### StackedGRU

A stacked GRU that processes the input over `n_steps` decoding iterations.
The input at each step is either repeated from the first step or zero-padded.

Key hyperparameters:

| Parameter | Description |
|---|---|
| `hidden_size` | GRU hidden state dimension |
| `n_layers` | Number of stacked GRU layers |
| `n_steps` | Number of decoding iterations |
| `dropout` | Dropout between layers |
| `expand_input` | Input mode per step: `"repeat"` or `"zero"` |

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

## Available Codes

Codes are stored as MATLAB `.mat` files under `data/codes/`, each containing a generator matrix `G` (k×n) and a parity-check matrix `H` (m×n):

| File | n | k | Rate |
|---|---|---|---|
| `bch.7.4.mat` | 7 | 4 | 0.571 |
| `bch.15.7.mat` | 15 | 7 | 0.467 |
| `bch.15.11.mat` | 15 | 11 | 0.733 |
| `bch.31.16.mat` | 31 | 16 | 0.516 |
| `bch.31.21.mat` | 31 | 21 | 0.677 |
| `bch.63.45.mat` | 63 | 45 | 0.714 |
| `bch.63.51.mat` | 63 | 51 | 0.810 |
| `ebch.32.11.mat` | 32 | 11 | 0.344 |
| `ebch.32.16.mat` | 32 | 16 | 0.500 |
| `ebch.32.21.mat` | 32 | 21 | 0.656 |
| `ebch.64.45.mat` | 64 | 45 | 0.703 |
| `ebch.128.64.mat` | 128 | 64 | 0.500 |
| `ldpc.tukl.96.48.mat` | 96 | 48 | 0.500 |

## Adding a New Code

Codes are defined by a MATLAB `.mat` file containing a generator matrix `G` (k×n) and a parity-check matrix `H` (m×n).
Point to it in your experiment config:

```yaml
code:
  mat_file: data/codes/my_code.mat
```

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
