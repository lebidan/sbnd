---
name: sbnd-eval
description: Evaluate a trained SBND model checkpoint with sbnd-test. Use when the user says "evaluate", "test", "benchmark", "compute WER/BER", or asks for a TTS / HDD eval. Resolves the checkpoint from any reasonable description (experiment name, wandb run name, date/time, "the one I just trained"), picks the matching eval preset, translates plain-English TTS/HDD requests into Hydra overrides, and parses the result CSV.
---

# sbnd-eval

Run `sbnd-test` for the user and summarize the resulting WER/BER sweep. The user should never need to paste a checkpoint path if the model is discoverable.

## Resolving the checkpoint

User input → ckpt path. Try these in order, stop at the first that yields exactly one match:

1. **Just-trained in this session** — if a `sbnd-train` run completed earlier in the conversation, prefer its ckpt. Always confirm if more than one was trained.
2. **Explicit path** — if it ends in `.ckpt` and the file exists, use it verbatim.
3. **Date / time anchor** — user says "the one from yesterday afternoon" / "trained around 14:30 on May 17". Glob `log/train/runs/<YYYY-MM-DD>_*/checkpoints/*.ckpt`, pick the run whose dir timestamp is closest to the named time (within ±60 min by default). If ambiguous, list the candidates with their exp name and ask.
4. **Experiment-name anchor** — user names an exp (`recct-bch-63-45-ml-4m-2dB-aug`) or a fragment ("the 63,45 rECCT"). Glob `log/train/runs/*/checkpoints/<fragment>*.ckpt`, sort by mtime, pick the newest if it's unambiguous; otherwise list and ask.
5. **Wandb run-name anchor** — user says "fiery-tree-1917". The wandb name is the trailing token of the ckpt filename. Glob `log/train/runs/*/checkpoints/*<wandb-name>*.ckpt`.

Skip `last.ckpt` unless the user explicitly asks for the latest training state rather than the best — `last.ckpt` is the periodic snapshot, the unsuffixed file (or `<exp>-<epochs>epochs[-<wandb>].ckpt`) is the best-val one.

Once resolved, **state the path you're using before launching** so the user can catch a wrong pick.

## Picking the eval preset

Read the code from the source experiment (either via the resolved `<output_dir>/.hydra/config.yaml` or by inferring from the exp filename) and map it to a `conf/eval/*.yaml`:

| code `mat_file` | `eval=` preset |
| --- | --- |
| `bch.31.21.mat` | `bch-31-21` |
| `bch.63.45.mat` | `bch-63-45` |
| `ebch.32.16.mat` | `ebch-32-16` |
| `rm.32.16.mat` | `rm-32-16` |
| `ldpc.rptu.96.48.mat` | `ldpc-rptu-96-48` |
| `ldpc.ccsds.128.64.mat` | `ldpc-ccsds-128-64` |
| `polar.rptu.128.64.mat` | `polar-rptu-128-64` |

If the code has no preset, omit `eval=` (the base `test.yaml` defaults are sane) and tell the user — they may want to override `snr_min/max/step`, `batch_size`, `num_batches`.

## Translating user phrasing into overrides

Build a single CLI from the patterns below. Always positional ckpt first.

**Hard-decision decoding (HDD) post-filter** (requires `error_space=codeword` + a code with known `dmin`):
```
hdd=true
```

**Self-Boosting TTS** ("with self-boosting N iters"):
```
tts._target_=sbnd.tts.SelfBoostingDecoder tts.num_iters=N
```

**Test-Time Augmentation TTS** ("with TTA N perms") — pick the transform from the code family:

| Code family | Transform |
| --- | --- |
| BCH / eBCH | `sbnd.transforms.BCHPerms` (no extra args) |
| QC-LDPC | `sbnd.transforms.QCPerms` (no extra args) |
| RM | `sbnd.transforms.GenericPerms` with `mat_file=${perms_dir}/perms.rm.32.mat`, `num_perms=1024` |
| Polar | `sbnd.transforms.GenericPerms` with `mat_file=${perms_dir}/perms.polar.128.mat`, `num_perms=1024` |

For BCH/QC the simple override block:
```
+tts._target_=sbnd.tts.TTADecoder \
+tts.num_perms=N \
+tts.transform._partial_=true \
+tts.transform._target_=sbnd.transforms.BCHPerms
```

For RM/Polar add the `mat_file` and `num_perms` to the transform block. Check `docs/evaluation.md` for canonical examples if you're unsure.

**SNR range tweaks**: `snr_min=`, `snr_max=`, `snr_step=`.
**Monte-Carlo budget**: `num_batches=`, `batch_size=`. Larger budget → tighter low-WER estimates. Defaults from the eval preset are usually fine.

## Launching

Always background. From the repo root:

```bash
.venv/bin/sbnd-test <ckpt> [eval=<preset>] [overrides...]
```

The positional ckpt is rewritten to `model=<path>` by an `sbnd-test` shim, so don't double-write `model=`. Eval is generally faster than training but can still take several minutes on big sweeps — use `Bash(run_in_background: true)` and `Monitor` the live log.

## Result CSV

Output goes to:

```
log/test/<ckpt-stem>[<tts-suffix>][-hdd].csv
```

Where `<ckpt-stem>` is the ckpt filename without `.ckpt`, and the TTS suffix is `-sbN` (self-boosting, N iters) or `-ttaN` (TTA, N perms) or empty. The file is **append + dedup by SNR**, so re-running with different `num_batches` overwrites in place.

After the run completes, read the CSV and report a compact per-SNR table (snr_db, fer, ber, n_words). Flag any SNR where `n_word_errors < ~100` — the FER estimate is noisy there.

## Out of scope

- Don't modify `conf/eval/*.yaml` from this skill. Tweaks belong as CLI overrides.
- Don't fall back to a stale memory-cached ckpt path without verifying the file still exists.
