# Evaluating a model

This document describes how to evaluate a trained SBND model with `sbnd-test`. It is organized in three sections: the basic Monte-Carlo SNR sweep to measure WER and BER, the optional hard-decision decoding (HDD) emulation used as a cheap post-filter, and the test-time scaling (TTS) variants that trade extra inference compute for lower error rates.

**See also:** [README](../README.md#-getting-started) · [Training a model](training.md) · [Extending SBND](extending.md)

## Contents

1. [Basic evaluation](#1-basic-evaluation)
   - [Output file](#output-file)
   - [Options](#options)
2. [Hard-decision decoding emulation](#2-hard-decision-decoding-emulation)
3. [Test-time scaling](#3-test-time-scaling)
   - [Self-boosting](#self-boosting)
   - [Test-time augmentation](#test-time-augmentation)
   - [Combining TTS with HDD](#combining-tts-with-hdd)

## 1. Basic evaluation

`sbnd-test` evaluates a trained checkpoint through Monte-Carlo simulation over a configurable range of Eb/N0 values, reporting **Word Error Rate (WER)** and **Bit Error Rate (BER)** at each SNR point. The decoding mode (`error_space`) used at training time is read back from the checkpoint, so WER/BER are computed accordingly — see [Decoding modes](../README.md#decoding-modes) in the README, or the table below for a quick reference.

| `error_space` | WER calculated on | BER calculated on |
| --- | --- | --- |
| `codeword` | decoded codeword | decoded message |
| `message` | decoded *message* | decoded message |

Like `sbnd-train`, evaluation is configured with [Hydra](https://hydra.cc/). The base config [`conf/test.yaml`](../conf/test.yaml) ships with sensible Monte-Carlo defaults, so a first evaluation pass requires only the model checkpoint:

```
sbnd-test model=/path/to/my-model.ckpt
```

where the checkpoint typically lives at `./log/train/runs/YYYY-MM-DD_HH-MM-SS/checkpoints/<exp-file-name>-<max_epochs>epochs-<wandb-run-name>.ckpt`.

Any field can be overridden directly on the command line:

```
sbnd-test model=/path/to/my-model.ckpt \
  snr_min=1 snr_max=5 snr_step=0.5 num_batches=8192 batch_size=4096
```

For repeated evaluations with the same set of options, group them into a preset under [`conf/eval/`](../conf/eval) and select it with `eval=<name>` (e.g. one preset per code):

```
sbnd-test model=/path/to/my-model.ckpt eval=my-eval-config
```

A few presets for the codes shipped with SBND are available in [`conf/eval/`](../conf/eval). You may need to adjust the batch size and number of batches to match your GPU.

### Output file

Results are saved to a CSV file named after the checkpoint, under the output directory (default: [`./log/test/`](../log/test)). If the file already exists, new SNR points are appended; for SNR points that are already present, error counts are **accumulated** on top of the previous ones (and WER/BER are recomputed from the cumulative totals). This makes it possible to extend an evaluation incrementally across multiple runs and progressively tighten the statistics.

The active TTS strategy and the HDD flag are reflected in the CSV filename suffix, so that different configurations of the same checkpoint do not overwrite one another (e.g. `<model>.csv`, `<model>-hdd.csv`, `<model>-sb5.csv`, `<model>-tta4-hdd.csv`).

### Options

| Option | Default | Description |
| --- | --- | --- |
| `model` | — (required) | Path to the model checkpoint to evaluate |
| `snr_min` / `snr_max` / `snr_step` | 0.0 / 5.0 / 1.0 | Eb/N₀ range to simulate (dB) |
| `batch_size` | 4096 | Test batch size |
| `num_batches` | 1024 | Number of batches per SNR point |
| `num_workers` | 8 | Number of workers for dataloading |
| `hdd` | `false` | Enable hard-decision decoding emulation — see §2 |
| `tts` | `SingleShotDecoder` | Decoding strategy — see §3 |
| `output_dir` | `./log/test` | Output directory for the results CSV |

## 2. Hard-decision decoding emulation

Setting `hdd=true` enables a hard-decision decoding emulation in which any prediction is declared successful as soon as the number of bit errors in the error pattern inferred by the SBND model is at most `t = ⌊(d_min − 1) / 2⌋`, the bounded-distance correction radius of the code. 

HDD emulation requires:

* a code with a known minimum distance `d_min` (loaded from the `.mat` file — see [Codes](../README.md#codes) in the README);
* a model trained with `error_space=codeword` (codeword-level error counting is required to evaluate the bounded-distance condition).

```
sbnd-test model=/path/to/my-model.ckpt eval=bch-31-21 hdd=true
```

Output: results are written to `<model>-hdd.csv`. HDD is orthogonal to TTS and may be combined with it — see [Combining TTS with HDD](#combining-tts-with-hdd).

## 3. Test-time scaling

Beyond the standard decoding mode (one forward pass per sample, the default), `sbnd-test` supports two test-time scaling (TTS) variants that exchange additional inference compute for lower error rates. The two variants are complementary: **self-boosting** is a sequential strategy in which the model iterates over its own predictions, while **test-time augmentation** is a parallel strategy in which the model is run on multiple equivalent views of each received word obtained via code automorphisms. Both are implemented in [`src/tts.py`](../src/tts.py).

Both TTS variants require a model trained in `error_space=codeword`, since they rely on the syndrome check `synd(ê) ≡ s_chan` to decide either when to terminate the loop (self-boosting) or which permuted prediction to keep (TTA). The active strategy is selected through the `tts:` block in the evaluation config, and Hydra-instantiated through `_target_`. The default is the no-TTS baseline, defined in [`conf/test.yaml`](../conf/test.yaml) as `_target_: sbnd.tts.SingleShotDecoder`.

### Self-boosting

In self-boosting (sequential TTS, [`SelfBoostingDecoder`](../src/tts.py)), the model iterates over its own predictions in an attempt to clean them up. The loop terminates as soon as a sample's prediction passes the syndrome check, or after `num_iters` model invocations, whichever comes first. A detailed description is provided in the [PhD thesis of A. Ismail, Chap. 4.2](https://theses.fr/2025IMTA0515). Early references to such a strategy are the *Iterative Error Correction* approach of [Kavvousanos & Paliouras, GLOBECOM 2020](https://ieeexplore.ieee.org/document/9367553) and the *Iterative Error Decimation* decoder by [Kamassury & Silva (2021)](https://arxiv.org/abs/2012.00089).

```yaml
tts:
  _target_: sbnd.tts.SelfBoostingDecoder
  num_iters: 5
```

```
sbnd-test model=/path/to/my-model.ckpt eval=bch-31-21 \
  tts._target_=sbnd.tts.SelfBoostingDecoder +tts.num_iters=10
```

Output: results are written to `<model>-sb<num_iters>.csv` (e.g. `<model>-sb5.csv`).

### Test-time augmentation

In test-time augmentation (parallel TTS, [`TTADecoder`](../src/tts.py)), the model is run independently on `num_perms` permuted versions of each received word. The permutations are drawn at random from the code's automorphism group, supplied via the same transform classes used for training-time data augmentation (`BCHPerms`, `QCPerms`, `GenericPerms` — see [Data augmentation](training.md#data-augmentation)). For each permutation, the resulting logits are inverse-permuted back into the original coordinate system. A sample stops as soon as one of its permutations yields a prediction that passes the syndrome check; for samples that never pass, the decoder output is obtained by averaging the predictions of all permutations.

```yaml
tts:
  _target_: sbnd.tts.TTADecoder
  num_perms: 4
  transform:
    _partial_: true
    _target_: sbnd.transforms.BCHPerms   # same classes as the training transform
    is_extended: false                    # set true for eBCH codes
```

The `_partial_: true` pattern lets Hydra inject the loaded `code` into the transform at decode time, mirroring the training-time data-augmentation setup.

```
sbnd-test model=/path/to/my-model.ckpt eval=bch-31-21 \
  tts._target_=sbnd.tts.TTADecoder +tts.num_perms=8 \
  +tts.transform._target_=sbnd.transforms.BCHPerms \
  +tts.transform._partial_=true
```

Output: results are written to `<model>-tta<num_perms>.csv` (e.g. `<model>-tta4.csv`).

### Combining TTS with HDD

The HDD flag is independent of the TTS strategy: it acts as a post-processing filter on the error counts and may be combined with any TTS variant. The two suffixes accumulate in the output filename, e.g. `<model>-sb5-hdd.csv` or `<model>-tta4-hdd.csv`.

```
sbnd-test model=/path/to/my-model.ckpt eval=bch-31-21 \
  hdd=true \
  tts._target_=sbnd.tts.SelfBoostingDecoder +tts.num_iters=5
```

### Practical considerations

The self-boosting and TTA strategies have been compared in the [PhD thesis of A. Ismail, Chap. 4.2](https://theses.fr/2025IMTA0515). Both rapidly increase the inference cost and show diminishing returns as the underlying model gets better. Whenever applicable, hard-decision decoding emulation (the `hdd` flag, §2) remains the most cost-efficient and effective strategy to get an extra boost in performance at inference time.
