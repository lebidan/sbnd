# sbnd — Syndrome-Based Neural Decoding

A PyTorch/Lightning framework for training and evaluating neural decoders for linear error-correcting codes.
Given a noisy received word, the decoder estimates the underlying error pattern by exploiting the code's syndrome and parity-check structure.

## Decoders

| Model | Description |
|---|---|
| `ECCT` | Transformer encoder with code-aware attention masking, adapted from [ECCT](https://github.com/yoniLc/ECCT) |
| `StackedGRU` | Stacked GRU decoder unrolled over multiple decoding steps |

Both decoders take as input the received symbol magnitudes `|y|` and the binary syndrome `s`, and output LLR estimates for the error pattern.

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone <repo-url>
cd sbnd

uv sync            # runtime dependencies
uv sync --group dev  # + black and mypy
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

## Adding a New Code

Codes are defined by a MATLAB `.mat` file containing a generator matrix `G` (k×n) and a parity-check matrix `H` (m×n).
Point to it in your experiment config:

```yaml
data:
  code_path: data/codes/my_code.mat
```
