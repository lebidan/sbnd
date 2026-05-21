# Agent guide for SBND

This file is the cross-agent entry point for any coding assistant working in this repo (Claude Code, Cursor, Codex CLI, Aider, etc.). It points at the canonical project docs and the per-workflow skill files.

For Claude-family agents, the same project conventions live in `CLAUDE.md`.

## Project at a glance

SBND is a PyTorch/Lightning framework for training and evaluating syndrome-based neural decoders for linear error-correcting codes. See `README.md` for the full overview.

- Source: `src/` (the `sbnd` Python package; entry points `sbnd-train` and `sbnd-test`).
- Configs: `conf/` (Hydra; `conf/exp/` experiment presets, `conf/eval/` evaluation presets).
- Data: `data/codes/` (code definitions, `.mat`), `data/perms/` (automorphism matrices).
- Outputs: `log/train/runs/<timestamp>/` for training runs, `log/test/*.csv` for eval sweeps.
- Env: local virtualenv at `.venv/` with `black`, `mypy`, `sbnd-train`, `sbnd-test` already installed.

**Invoking venv binaries.** Always call console scripts by their absolute venv path — `.venv/bin/sbnd-train`, `.venv/bin/sbnd-test`, `.venv/bin/black`, `.venv/bin/mypy`. The shebang inside each shim points at `.venv/bin/python`, so the venv's site-packages (PyTorch, Lightning, Hydra, …) load automatically and child processes inherit `sys.executable`. Do **not** `source .venv/bin/activate` from a skill, and do not rely on the user having activated the venv before launching the agent.

## Documentation (source of truth)

- `README.md` — overview, install, quickstart.
- `docs/training.md` — `sbnd-train` reference.
- `docs/evaluation.md` — `sbnd-test` reference.
- `docs/extending.md` — adding a decoder via `BaseDecoder`.
- `docs/experiments.md` — index of `conf/exp/` configs cross-referenced with `log/test/` results.

Keep these in sync when changing user-visible behavior, CLI flags, config schema, or the decoder API.

## Common workflows — skill files

Workflow recipes for common dialog-driven tasks live under `skills/<name>/SKILL.md` at the repo root. They use the Anthropic Agent Skills format (YAML frontmatter + markdown body) but are plain markdown and readable by any agent.

- `skills/sbnd-train/SKILL.md` — kick off a training run, resolve `exp=` preset, apply overrides, return the resulting checkpoint path.
- `skills/sbnd-eval/SKILL.md` — evaluate a checkpoint with `sbnd-test`; resolves the ckpt from any reasonable description (exp name, wandb run name, date/time, "the one I just trained"), picks the matching eval preset, translates plain-English TTS/HDD requests into Hydra overrides.
- `skills/sbnd-new-exp/SKILL.md` — scaffold a new `conf/exp/*.yaml` config from a description, using an existing preset as a template, then hand off to training.

Agents that support automatic skill loading (Claude Code, Claude.ai, Agent SDK) will invoke these by name. Agents that don't can still read them as workflow documentation.

## When modifying source code

1. Write a small smoke test for the change. For training-touching changes, the `conf/exp/dev-test-*.yaml` presets are good starting points — they finish in seconds.
2. Run `.venv/bin/black src/` and `.venv/bin/mypy src/`; both must be clean.
3. Update the relevant `docs/*.md` (and `README.md` if needed) for user-visible changes.
4. There is no test suite yet — rely on smoke tests + type checks.

## Path & naming conventions

- Training output dir: `log/train/runs/<YYYY-MM-DD_HH-MM-SS>/`
- Best ckpt: `<output_dir>/checkpoints/<exp>-<max_epochs>epochs[-<wandb-run-name>].ckpt`
- Periodic ckpt: `<output_dir>/checkpoints/last.ckpt`
- Eval CSV: `log/test/<ckpt-stem>[<tts-suffix>][-hdd].csv` (`-sbN` for self-boosting, `-ttaN` for TTA, appended in place on re-run)

These conventions are exploited by the skill files for checkpoint discovery — keep them stable.
