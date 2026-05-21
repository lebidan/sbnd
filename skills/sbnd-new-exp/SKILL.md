---
name: sbnd-new-exp
description: Scaffold a new conf/exp/*.yaml experiment config from a description. Use when the user wants to train a combination (decoder × code × data × SNR × epochs) that no existing preset covers. Picks the closest existing preset as a template, applies the requested changes, writes the new file, and hands off to sbnd-train.
---

# sbnd-new-exp

Author a new Hydra experiment config under `conf/exp/`. The goal is a minimal diff from an existing, known-good preset — not a config written from scratch.

## Process

1. **Parse the request** into a tuple: `(decoder, code, data_strategy, training_snr, epochs, augmentation?)`.
   - Decoder: `gru`, `ecct`, `crossmpt`, `recct`, or a new decoder class in `src/`.
   - Code: matches the `data/codes/*.mat` family (`bch.31.21`, `bch.63.45`, `ebch.32.16`, `rm.32.16`, `ldpc.rptu.96.48`, `ldpc.ccsds.128.64`, `polar.rptu.128.64`).
   - Data: `on-demand` or `ml-<size>` (pre-computed ML dataset, e.g. `ml-1m`, `ml-4m`, `ml-16m`).
   - SNR: training Eb/N0 in dB.
   - `-aug` if code automorphisms are used for augmentation.

2. **Pick the closest template** by globbing `conf/exp/*.yaml` and ranking matches in this priority: same decoder → same code → same data strategy → same SNR. Read the template fully before diffing.

3. **Propose the new filename** following the existing convention:
   ```
   <decoder>-<code>-<data>-<snr>dB[-aug].yaml
   ```
   e.g. `recct-ebch-32-16-ml-4m-3dB-aug.yaml`. Confirm with the user before writing.

4. **Apply the diff**, touching only what the request requires:
   - `decoder` block (`_target_`, `embed_dim`, `n_heads`, `n_layers`, `n_iters`, dropouts).
   - `code.mat_file` → matching `data/codes/<...>.mat`.
   - `data` block: `ebno_dB_train`, `n_train_samples`, `n_val_samples`, `ebno_dB_test`, `n_test_samples`, plus `train_files` / `val_files` for pre-computed sets.
   - `max_epochs`, `lr`, `optimizer`, `lr_scheduler`.
   - Augmentation transform if `-aug` (decoder's data transform block).

5. **Write the file** with `Write`. Keep the same section headers, comment style, and 80-col wrapping as the template — diff-friendliness matters here.

6. **Hand off**: report the path, summarize the diff vs. the template, and offer to launch via `sbnd-train`.

## Hard rules

- Never invent a `data/codes/*.mat` file. If the code is new, ask the user where the `.mat` lives — don't fabricate a path.
- Never invent a pre-computed dataset path. `data/datasets/` is gitignored; ask before referencing a new file.
- Preserve `${codes_dir}`, `${data_dir}`, `${perms_dir}` variable refs — don't inline them.
- Do not touch `conf/train.yaml` or `conf/test.yaml` from this skill.
- After writing, run `.venv/bin/sbnd-train exp=<new-name> trainer.fast_dev_run=true` as a sanity check before launching the real run — config errors are cheap to surface this way.

## Out of scope

- New decoder architectures (subclasses of `BaseDecoder`) — that's a `src/` change, see `docs/extending.md`.
- New eval presets — usually unnecessary; eval presets are per-code, not per-experiment.
