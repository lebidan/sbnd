# Track A (input embedding) — running notes

Branch: `exp/track-a-embedding`. Plan: `exp/recct_experiment_plan.md`.
Last updated: 2026-09-17.

## What is being tested

ECCT/rECCT embeds its input multiplicatively: `x_i = value_i * e_i`. The encoder
is pre-norm and `LayerNorm(c * e_i)` is independent of `c > 0`, so the first
attention sub-layer sees position and syndrome sign but no graded reliability.
Track A replaces this with additive forms (`position + content`).

Measured while building the self-check: the invariance is exact only as
`eps -> 0`. At the repo's `std=0.02` init a token vector's variance is ~1e-4, so
the default LayerNorm `eps=1e-5` is ~10% of it and leaks a little reliability —
strongly nonlinear and concentrated at `|y| ~ 0`, i.e. a degenerate "this bit is
near-zero reliability" signal rather than graded information.

## Variants (`embedding` / `embed_feat` / `embed_syn` on `sbnd.recct.RECCT`)

| id  | bits                     | syndrome                  | args |
|-----|--------------------------|---------------------------|------|
| A0  | `v_i * \|y_i\|` (mult)     | `v_j * s_j` (mult)        | `embedding=mult` (default) |
| A1  | `p_i + v_i * \|y_i\|`      | `p_j + v_j * s_j`         | `embedding=affine` |
| A2a | `p_i + MLP(f(\|y_i\|))`    | `p_j + syn[b]` (2 shared) | `embedding=decoupled embed_feat=raw` |
| A2r | `p_i + MLP(f(\|y_i\|))`    | `p_j + v_j * s_j`         | `embedding=decoupled embed_feat=raw embed_syn=affine` |

A2a changes both halves at once, so it cannot be attributed on its own; A2r
isolates the reliability change relative to A1. `embed_syn=affine` keeps the
per-check direction `v_j` (m extra DoF); `symbols` forces the same displacement
`syn[1]-syn[0]` on every check.

## Protocol (fixed across all runs — do not vary mid-series)

- Same hyperparameters for every variant; parameter counts are NOT matched.
- Seed 1234, `gpus=4 cpus=12` (DDP). `data.train_bs` is the GLOBAL batch size —
  `src/train.py` divides it by the GPU count. The rank count must stay fixed,
  since data ordering differs between 1 and 4 ranks.
- W&B online (never `offline=true` outside smoke tests).
- Judgment is on the whole FER-vs-Eb/N0 curve (trend), not a single number.
- Plot with `exp/plot_fer.py` (overlay + ratio panel vs the first curve).

## Phase 1 — eBCH(32,16), DONE

`recct-ebch-32-16-ml-4m-3dB-aug` (4M ML + BCHPerms aug, 3 dB), `max_epochs=128`,
~65 min/run. Eval 1.0–6.0 dB step 1.0, 33.55M cw/point.

| id  | W&B run            | params  | FER @2 dB | @4 dB    | @6 dB    |
|-----|--------------------|---------|-----------|----------|----------|
| A0  | fancy-shadow-2101  | 154,016 | 8.240e-2  | 3.277e-3 | 1.025e-5 |
| A1  | frosty-glitter-2104| 158,624 | 8.114e-2  | 3.178e-3 | 9.537e-6 |
| A2r | olive-durian-2106  | 164,864 | 8.393e-2  | 3.282e-3 | 1.010e-5 |
| A2a | (not evaluated)    | 163,520 | worse than A0 on val/acc and post-fit test |

Result: **A1 < A0 < A2r < A2a**. A1 beats A0 at every SNR, by a margin growing
monotonically 1.1% -> 7.0% with Eb/N0 — consistent, but only ~0.01-0.02 dB.
A2r is worse than A0 at low SNR. Decomposition from the like-for-like post-fit
tests (2 dB): affine->MLP bits costs +3.6%, affine->symbol syndrome costs +2.3%.

Caveat: one seed, and A1 carries +3.0% parameters, so "small real effect" and
"capacity" are not separable at this effect size. Against pure capacity: A1's
train/val gap is *narrower* than A0's (-0.0042 vs -0.0054), and the gain is
monotonic in SNR, which capacity alone does not predict.

Figure: `exp/fer-ebch-32-16-trackA.png`.

## Phase 2 — LDPC RPTU (96,48,10), TODO

Question: does A1's advantage over A0 grow on a longer, harder code? This code
has sparse H (6.4% density vs eBCH's 30%), 144 tokens vs 48, and rECCT barely
matches BP-100 on it, so any real gain matters.

Base config `recct-ldpc-rptu-96-48-on-demand-3dB` (on-demand data at 3 dB, no
augmentation, `embed_dim=192 n_heads=8 n_iters=10`, lr 1e-4), with:

- `max_epochs=64` (full 512 is 1d6h on 4 GPUs; 64 epochs is ~3.7 h/run)
- `lr_scheduler.warmup=10 lr_scheduler.decay=16`

Note this is a COMPLETE compressed WSD schedule (LR annealed by epoch 64), not
the first 64 epochs of the 512-epoch schedule. These numbers are therefore NOT
comparable to the stored 512-epoch reference `zany-star-1923`.

Variants: A0, A1, A2r (skip A2a). Sequential, one at a time.
Eval: 1.0–5.0 dB step 1.0, `batch_size=8192 num_batches=4096` (33.55M cw/point).

## Known repo gotchas hit along the way

- `dev-test-*` presets set `offline: true`, which masks any W&B-dependent bug.
  Smoke tests of W&B behavior need an explicit `offline=false`.
- Overriding `max_epochs` below `warmup + decay` trips an assert in
  `sbnd.lr_sched.WarmupStableDecayLR`.
- Fixed on main (`8435f66`): `WandbModifyCheckpointName` read
  `logger.experiment.name` on every DDP rank, but only rank 0 has a real
  `wandb.Run`; the others got a `_DummyExperiment` bound method baked into the
  checkpoint filename, which broke `trainer.test(ckpt_path="best")` after fit.
