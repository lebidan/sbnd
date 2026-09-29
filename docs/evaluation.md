# Evaluating a model

This document describes how to evaluate a trained SBND model with `sbnd-test`. It is organized in four sections: the basic Monte-Carlo SNR sweep to measure WER and BER, the optional hard-decision decoding (HDD) emulation used as a cheap post-filter, the test-time scaling (TTS) variants that trade extra inference compute for lower error rates, and the iteration-count override for iterative decoders.

**See also:** [README](../README.md#-getting-started) · [Training a model](training.md) · [Extending SBND](extending.md) · [Experiments](experiments.md)

## Contents

1. [Basic evaluation](#1-basic-evaluation)
   - [Output file](#output-file)
   - [Options](#options)
2. [Hard-decision decoding emulation](#2-hard-decision-decoding-emulation)
3. [Test-time scaling](#3-test-time-scaling)
   - [Self-boosting](#self-boosting)
   - [Test-time augmentation](#test-time-augmentation)
   - [AfterBurner decoding](#afterburner-decoding)
   - [Combining TTS with HDD](#combining-tts-with-hdd)
4. [Overriding the iteration count](#4-overriding-the-iteration-count)

## 1. Basic evaluation

`sbnd-test` evaluates a trained checkpoint through Monte-Carlo simulation over a configurable range of Eb/N0 values, reporting **Word Error Rate (WER)** and **Bit Error Rate (BER)** at each SNR point. The decoding mode (`error_space`) used at training time is read back from the checkpoint, so WER/BER are computed accordingly — see [Decoding modes](../README.md#decoding-modes) in the README, or the table below for a quick reference.

| `error_space` | WER calculated on | BER calculated on |
| --- | --- | --- |
| `codeword` | decoded codeword | decoded message |
| `message` | decoded *message* | decoded message |

Like `sbnd-train`, evaluation is configured with [Hydra](https://hydra.cc/). The base config [`conf/test.yaml`](../conf/test.yaml) ships with sensible Monte-Carlo defaults, so a first evaluation pass requires only the model checkpoint:

```
sbnd-test /path/to/my-model.ckpt
```

where the checkpoint typically lives at `./log/train/runs/YYYY-MM-DD_HH-MM-SS/checkpoints/<exp-file-name>-<max_epochs>epochs-<wandb-run-name>.ckpt`. The first positional argument is rewritten to `model=<path>` for Hydra, meaning the explicit `model=/path/to/my-model.ckpt` form is also valid.

Any field can be overridden directly on the command line:

```
sbnd-test /path/to/my-model.ckpt \
  snr_min=1 snr_max=5 snr_step=0.5 num_batches=8192 batch_size=4096
```

For repeated evaluations with the same set of options, group them into a preset under [`conf/eval/`](../conf/eval) and select it with `eval=<name>` (e.g. one preset per code):

```
sbnd-test /path/to/my-model.ckpt eval=my-eval-config
```

A few presets for the codes shipped with SBND are available in [`conf/eval/`](../conf/eval). You may need to adjust the batch size and number of batches to match your GPU.

### Output file

Results are saved to a CSV file named after the checkpoint, under the output directory (default: [`./log/test/`](../log/test)). If the file already exists, new SNR points are appended; for SNR points that are already present, error counts are **accumulated** on top of the previous ones (and WER/BER are recomputed from the cumulative totals). This makes it possible to extend an evaluation incrementally across multiple runs and progressively tighten the statistics.

Rows are written sorted by Eb/N0. Each run only adds its own counts to what is on disk, under an exclusive file lock, so several `sbnd-test` processes can write the same output file concurrently (e.g. one per GPU, on the same or different SNR points) and their counts add up. The lock is a `<csv>.lock` sidecar file created next to the CSV. **Caveat:** on cluster filesystems where `flock` is node-local (e.g. Lustre mounted with `localflock`), concurrent runs are only safe when they run on the same node.

The active TTS strategy, an `n_iters` override, the precision and the HDD flag are reflected in the CSV filename suffix, in the order `<model>[<tts>][-it<n>][-bf16][-hdd].csv`, so that different configurations of the same checkpoint do not overwrite one another (e.g. `<model>.csv`, `<model>-hdd.csv`, `<model>-sb5.csv`, `<model>-ab6.csv`, `<model>-bf16.csv`, `<model>-it20-bf16.csv`, `<model>-tta4-bf16-hdd.csv`).

### Options

| Option | Default | Description |
| --- | --- | --- |
| `model` | — (required) | Path to the model checkpoint to evaluate |
| `snr_min` / `snr_max` / `snr_step` | 0.0 / 5.0 / 1.0 | Eb/N₀ range to simulate (dB) |
| `batch_size` | 4096 | Test batch size |
| `num_batches` | 1024 | Number of batches per SNR point (a maximum when `min_cw_errors > 0`) |
| `min_cw_errors` | 500 | Stop an SNR point early once this run has seen this many codeword errors; `0` = always run `num_batches` — see below |
| `num_workers` | 2 | Number of workers for dataloading (must be >= 1, the same rule as for training) |
| `precision` | `32-true` | `32-true` (fp32) or `bf16-mixed` (bf16 autocast, adds `-bf16` to the CSV name) — see below |
| `n_iters` | `null` | Evaluate an iterative decoder at this many iterations, with syndrome-based early exit (adds `-it<n>` to the CSV name) — see §4 |
| `hdd` | `false` | Enable hard-decision decoding emulation — see §2 |
| `tts` | `SingleShotDecoder` | Decoding strategy — see §3 |
| `output_dir` | `./log/test` | Output directory for the results CSV |

**Early stop on error count.** The accuracy of a Monte Carlo WER estimate depends on the number of errors observed (relative std ≈ 1/√errors), not on the number of words simulated: 500 errors give a 95% confidence interval of about ±9% on the WER, 1000 errors about ±6%. With `min_cw_errors=N`, each SNR point stops at the first batch where the run's codeword errors reach `N`, so low-SNR points finish quickly and `num_batches` only caps the high-SNR ones. The count covers the current run only (not what is already in the CSV): re-running a point always adds ≥ `N` new errors, and `k` concurrent runs on the same file yield ~`k × N` errors. Stopping on the error count biases the WER by ~1/`N` relative, negligible next to the statistical noise.

**Precision and compilation.** Models trained with `precision: bf16-mixed` should be evaluated with `precision=bf16-mixed`: evals run much faster (about 2.5× on BCH(31,21) RECCT) with the same WER. The fp32 default (`32-true`) is always safe, and some models (e.g. GRU) need it. In both precisions, the decoder is `torch.compile`d when its checkpoint was trained with `compile: true`; the first batch then pays a few seconds of compile warm-up.

**Zero-syndrome words.** Received words with a zero syndrome are not fed to the model (at high SNR they are most of the batch): their decoded word is the hard decision, left uncorrected. They are still counted in `Total CW`; those with a nonzero error pattern (undetectable errors, the error being a codeword) are counted as codeword errors, and their bit errors are the hard-decision ones.

**CPU threads.** `sbnd-test` deliberately runs its main process single-threaded on the CPU (its CPU work is only per-batch error counting), so setting `OMP_NUM_THREADS` is not needed. On Slurm, request `num_workers + 1` CPUs (3 by default, e.g. `--cpus-per-task=3`): the cluster bills allocated CPUs, so the savings only show up if the allocation shrinks too. Without a GPU the model runs on the CPU and all threads are kept.

## 2. Hard-decision decoding emulation

Setting `hdd=true` enables a hard-decision decoding emulation in which any prediction is declared successful as soon as the number of bit errors in the error pattern inferred by the SBND model is at most `t = ⌊(d_min − 1) / 2⌋`, the bounded-distance correction radius of the code. 

HDD emulation requires:

* a code with a known minimum distance `d_min` (loaded from the `.mat` file — see [Codes](../README.md#codes) in the README);
* a model trained with `error_space=codeword` (codeword-level error counting is required to evaluate the bounded-distance condition).

```
sbnd-test /path/to/my-model.ckpt eval=bch-31-21 hdd=true
```

Output: results are written to `<model>-hdd.csv`. HDD is orthogonal to TTS and may be combined with it — see [Combining TTS with HDD](#combining-tts-with-hdd).

## 3. Test-time scaling

Beyond the standard decoding mode (one forward pass per sample, the default), `sbnd-test` supports three test-time scaling (TTS) variants that exchange additional inference compute for lower error rates. **Self-boosting** is a sequential strategy in which the model iterates over its own predictions, **test-time augmentation** is a parallel strategy in which the model is run on multiple equivalent views of each received word obtained via code automorphisms, and **AfterBurner decoding** is a list strategy in which the model re-decodes several bit-flip hypotheses, with saturated reliabilities, on the least reliable positions. All three are implemented in [`src/tts.py`](../src/tts.py).

All TTS variants require a model trained in `error_space=codeword`, since they rely on the syndrome check `synd(ê) ≡ s_chan` to decide either when to terminate the loop (self-boosting), which permuted prediction to keep (TTA), or which hypotheses are valid candidates (AfterBurner). The active strategy is selected through the `tts:` block in the evaluation config, and Hydra-instantiated through `_target_`. The default is the no-TTS baseline, defined in [`conf/test.yaml`](../conf/test.yaml) as `_target_: sbnd.tts.SingleShotDecoder`.

### Self-boosting

In self-boosting (sequential TTS, [`SelfBoostingDecoder`](../src/tts.py)), the model iterates over its own predictions in an attempt to clean them up. The loop terminates as soon as a sample's prediction passes the syndrome check, or after `num_iters` model invocations, whichever comes first. A detailed description is provided in the [PhD thesis of A. Ismail, Chap. 4.2](https://theses.fr/2025IMTA0515). Early references to such a strategy are the *Iterative Error Correction* approach of [Kavvousanos & Paliouras, GLOBECOM 2020](https://ieeexplore.ieee.org/document/9367553) and the *Iterative Error Decimation* decoder by [Kamassury & Silva (2021)](https://arxiv.org/abs/2012.00089).

```yaml
tts:
  _target_: sbnd.tts.SelfBoostingDecoder
  num_iters: 5
```

```
sbnd-test /path/to/my-model.ckpt eval=bch-31-21 \
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

For codes whose automorphisms are not directly captured by `BCHPerms` or `QCPerms`, use `GenericPerms` with a `.mat` file. Two example files are shipped under [`data/perms/`](../data/perms) (RM-32 and Polar-128); see [Data augmentation](training.md#data-augmentation) in the training guide for the file format:

```yaml
tts:
  _target_: sbnd.tts.TTADecoder
  num_perms: 4
  transform:
    _partial_: true
    _target_: sbnd.transforms.GenericPerms
    mat_file: ${perms_dir}/perms.rm.32.mat
```

The `_partial_: true` pattern lets Hydra inject the loaded `code` into the transform at decode time, mirroring the training-time data-augmentation setup.

```
sbnd-test /path/to/my-model.ckpt eval=bch-31-21 \
  tts._target_=sbnd.tts.TTADecoder +tts.num_perms=8 \
  +tts.transform._target_=sbnd.transforms.BCHPerms \
  +tts.transform._partial_=true
```

Output: results are written to `<model>-tta<num_perms>.csv` (e.g. `<model>-tta4.csv`).

### AfterBurner decoding

In AfterBurner decoding (list TTS, [`AfterBurnerDecoder`](../src/tts.py)), the model first decodes each received word once. Words whose prediction passes the syndrome check are kept as-is. For the others, the `num_flips` = p least reliable positions (smallest |y|) are selected and all 2^p flip patterns f over them (f = 0 included) are tested: for each pattern, the model is run on the syndrome of the flipped word, with the reliabilities of all p test positions set to 1.0 (every hypothesised bit value is frozen, flipped or not), and the candidate error pattern is the model's hard decision XOR f. Among the candidates that pass the syndrome check, the one minimising the ML metric Σ |y_i|·ê_i is kept; if none passes, the first-pass prediction is returned. This is the AfterBurner scheme of [S. Scholl, P. Schläfer, N. Wehn, *Saturated Min-Sum Decoding: An "Afterburner" for LDPC Decoder Hardware*, DATE 2016](https://ieeexplore.ieee.org/document/7459497), which runs after LDPC decoding fails, saturates the reliabilities of the least reliable positions in a Chase-like enumeration of hypotheses, and runs another LDPC decoding round on each perturbed word; here BP is replaced by the SBND model.

Since the hypotheses are only tested on detected failures, the cost is about 1 + FER·2^p forward passes per word, which falls quickly as the SNR increases. The hypotheses are batched and run in slices no larger than the evaluation batch, so peak memory stays that of a regular batch.

```yaml
tts:
  _target_: sbnd.tts.AfterBurnerDecoder
  num_flips: 6
```

```
sbnd-test /path/to/my-model.ckpt eval=ldpc-rptu-96-48 \
  tts._target_=sbnd.tts.AfterBurnerDecoder +tts.num_flips=6
```

Other rules for setting the reliabilities of the test positions were investigated on LDPC RPTU(96,48) with rECCT A1 (64 epochs), at 3 dB, over 2^20 words (53,304 first-pass failures, FER 5.08e-2 without AfterBurner). With p = 6, leaving the reliabilities unchanged gives FER 3.33e-2, setting only the flipped bits to 1.0 gives 2.00e-2, and setting all p test positions to 1.0 gives 8.65e-3. The last rule is about 0.4 dB better than flipped-only, and is the one implemented.

Output: results are written to `<model>-ab<num_flips>.csv` (e.g. `<model>-ab6.csv`).

### Combining TTS with HDD

The HDD flag is independent of the TTS strategy: it acts as a post-processing filter on the error counts and may be combined with any TTS variant. The two suffixes accumulate in the output filename, e.g. `<model>-sb5-hdd.csv`, `<model>-tta4-hdd.csv` or `<model>-ab6-hdd.csv`.

```
sbnd-test /path/to/my-model.ckpt eval=bch-31-21 \
  hdd=true \
  tts._target_=sbnd.tts.SelfBoostingDecoder +tts.num_iters=5
```

### Practical considerations

The self-boosting and TTA strategies have been compared in the [PhD thesis of A. Ismail, Chap. 4.2](https://theses.fr/2025IMTA0515). Both rapidly increase the inference cost and show diminishing returns as the underlying model gets better. Whenever applicable, hard-decision decoding emulation (the `hdd` flag, §2) remains the most cost-efficient and effective strategy to get an extra boost in performance at inference time.

## 4. Overriding the iteration count

Iterative decoders loop one weight-tied block a fixed number of times `T_train` during training: [`RECCT`](../src/recct.py) (its `n_iters`) and [`StackedGRU`](../src/gru.py) (its `n_steps`). Passing `n_iters` to `sbnd-test` evaluates the trained model at any other count `T'`, lower or higher, without retraining:

```
sbnd-test model=/path/to/my-model.ckpt n_iters=20
```

**Early exit on the syndrome is always on with this option.** Every frame runs all `T'` iterations, and the model's readout is applied after each one. The output of a frame is the readout of the first iteration whose hard decision satisfies the syndrome (its syndrome equals the channel syndrome). A frame that never satisfies it gets the readout at iteration `min(T', T_train)`, not the last one: that answer is a frame error either way, but running past the trained count can add bit errors, so this fallback keeps the BER from getting worse too.

At `T' >= T_train`, FER and BER therefore cannot get worse than evaluating at the trained count. At `T' < T_train` they can, since fewer iterations are run. `n_iters=<T_train>` gives the early-exit result at the trained count.

Without `n_iters`, the decoder runs its trained count with no early exit, exactly as before this option existed, so older CSVs stay reproducible.

Results go to a `-it<n>`-suffixed CSV (e.g. `<model>-it20.csv`, early exit implied), so a sweep over several counts does not merge rows into one file. The option combines with TTS, HDD and `precision`.

**Caveats:**

- The readout was only trained on the state reached after `T_train` iterations. Its answers at other iterations can be poor, and early exit only accepts the ones that satisfy the syndrome.
- The option only applies to `RECCT` and `StackedGRU` models trained with `error_space=codeword` (the early exit needs a codeword-space prediction to check the syndrome). Any other decoder or a `message`-space model raises an error.
- For a multi-layer `RECCT` (`n_layers > 1`), `n_iters` sets the loop count of every layer, and only the iterations of the last layer's loop are candidates: the earlier layers' states are not final answers.
