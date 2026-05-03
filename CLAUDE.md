# SBND — Syndrome-Based Neural Decoding

PyTorch/Lightning framework for training and evaluating syndrome-based neural decoders for linear error-correcting codes.

## Documentation

These are the source of truth — read them rather than rely on guesses, and keep them up to date when changing user-visible behavior, config schema, CLI flags, or the decoder API:

- `README.md` — overview, install, quickstart, project structure, supported codes/decoders.
- `docs/training.md` — `sbnd-train` reference.
- `docs/evaluation.md` — `sbnd-test` reference.
- `docs/extending.md` — how to add a new decoder via `BaseDecoder`.

## Layout

- `src/` — the `sbnd` Python package (entry points: `sbnd-train` → `src/train.py`, `sbnd-test` → `src/test.py`).
- `conf/` — Hydra configs (`train.yaml` / `test.yaml` bases, `exp/` and `eval/` presets).
- `data/codes/` — example code definition `.mat` files, ready to use for any test or experiment.
- `data/perms/` — code automorphism matrices.
- `docs/` — reference documentation (see above).

## Environment

- Local virtualenv at `.venv/` with all dependencies installed, including dev tools `black` and `mypy` (configured in `pyproject.toml`).
- No test suite yet.

## When modifying source code

1. Write a small smoke test appropriate to the change to confirm it works. If training is needed, prefer on-demand data; the `conf/exp/dev-test-*.yaml` experiments are good starting points / inspiration when working on a model.
2. Run `.venv/bin/black src/` and `.venv/bin/mypy src/`; both must be clean.
3. Update the relevant `docs/*.md` (and `README.md` if needed) when the change is user-visible.
