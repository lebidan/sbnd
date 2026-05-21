---
name: sbnd-train
description: Launch and monitor an SBND training run. Use when the user says "train", "fit", "continue training", "resume", or asks to start a new experiment. Resolves the right exp= preset, applies overrides, runs sbnd-train in the background, and reports the resulting checkpoint path.
---

# sbnd-train

Run `sbnd-train` for the user, picking the right `exp=<name>` preset and handing the resulting checkpoint to follow-up steps (typically `sbnd-eval`).

## Resolving the experiment

The user names an experiment in one of these forms:
- An exact preset name (`recct-bch-63-45-ml-4m-2dB-aug`) → use as-is.
- A description ("rECCT on the (63,45) BCH at 2 dB with 4M ML samples + aug") → list matching files under `conf/exp/*.yaml`, confirm with the user before launching. Naming pattern is `<decoder>-<code>-<data>-<snr>[-aug].yaml`.
- "Like X but Y" → if Y is a small override (epochs, lr, snr), reuse exp X and pass overrides on the CLI. If Y is structural (different model/code/data), hand off to the `sbnd-new-exp` skill.

If nothing matches, do **not** invent an `exp=` name — either propose `sbnd-new-exp` or ask.

## Common overrides

Pass on the CLI after `exp=<name>`:

| Intent | Override |
| --- | --- |
| Epoch count | `max_epochs=128` |
| Learning rate | `lr=5e-4` |
| Continue from a pre-trained ckpt (fresh optimizer) | `+continue=<ckpt>` |
| Resume an interrupted run | `+resume=<ckpt>` |
| Disable wandb online sync | `offline=true` |
| Different GPU count | `gpus=2` |
| Different num workers | `cpus=16` |
| Quick smoke test | `trainer.fast_dev_run=true` |

For `+continue=` / `+resume=`, resolve the ckpt the same way `sbnd-eval` does (see that skill). Don't ask the user to paste the path if it's discoverable.

## Launching

Always run in the background — training takes minutes to hours.

```bash
.venv/bin/sbnd-train exp=<name> [overrides...]
```

Use `Bash(run_in_background: true)`. The first lines of stdout (and the Hydra log header) print the output directory `log/train/runs/<YYYY-MM-DD_HH-MM-SS>/`. Capture it — that's your handle for the rest of the session.

For quick progress visibility while it runs, use `Monitor` on `<output_dir>/train.log` (one line per Lightning epoch).

## Checkpoint path convention

After completion, the best checkpoint lives at:

```
log/train/runs/<timestamp>/checkpoints/<exp>-<max_epochs>epochs[-<wandb-run-name>].ckpt
```

and `last.ckpt` sits beside it. The wandb suffix is appended by a custom callback once the wandb run name is known (e.g. `…-fiery-tree-1917.ckpt`); in `offline=true` mode the suffix is omitted.

## Reporting back

When training finishes, do three things:
1. Report the resolved ckpt path (glob `<output_dir>/checkpoints/*.ckpt`, prefer the non-`last.ckpt` one).
2. Surface the end-of-training test metrics — `sbnd-train` calls `trainer.test` after `trainer.fit`; the per-SNR WER/ACC lines are at the bottom of `train.log`.
3. Offer to run `sbnd-eval` on the new ckpt.

## Out of scope

- Don't edit `conf/exp/*.yaml` from this skill — use `sbnd-new-exp` for that.
- Don't touch `src/` — if training fails because of a code bug, report it and stop.
- Don't run `black`/`mypy` here; those belong to source-modification flows.
