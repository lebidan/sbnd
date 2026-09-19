# Track A (input embedding) — running notes

Branch: `exp/track-a-embedding`. Plan: `exp/recct_experiment_plan.md`.
Last updated: 2026-09-19.

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

## Phase 2 — LDPC RPTU (96,48,10), DONE

`recct-ldpc-rptu-96-48-on-demand-3dB` + variants, `max_epochs=64`,
`warmup 10 / decay 16` (a COMPLETE compressed WSD schedule, so NOT comparable to
the 512-epoch reference `zany-star-1923`), ~2h50/run. Eval 1.0-5.0 dB step 1.0,
33.55M cw/point.

| id  | W&B run             | val/loss | val/acc | FER @3 dB | @4 dB    | @5 dB    |
|-----|---------------------|----------|---------|-----------|----------|----------|
| A0  | generous-glitter-2107 | 0.0108 | 0.946   | 5.357e-2  | 3.842e-3 | 8.026e-5 |
| A1  | lively-resonance-2111 | 0.0103 | 0.949   | 5.127e-2  | 3.528e-3 | 6.837e-5 |
| A2r | swept-paper-2112      | 0.0104 | 0.949   | 5.074e-2  | 3.492e-3 | 6.896e-5 |

FER relative to A0: A1 0.989 / 0.976 / 0.957 / 0.918 / 0.852 and A2r 0.985 /
0.969 / 0.947 / 0.909 / 0.859 at 1/2/3/4/5 dB. Both additive variants beat the
baseline at every SNR, with the margin growing monotonically in Eb/N0 — the same
shape as phase 1 but about twice the size (15% at 5 dB vs 7% at 6 dB on eBCH).
In dB this is still only ~0.04 dB at 5 dB, where the curve falls 1.68 decades/dB.

Unlike phase 1, A2r is NOT worse than A0 here: A1 and A2r are within each other's
noise at every point. Whatever the per-check syndrome direction `v_j` was buying
on the dense eBCH parity check matrix (30% density) does not matter on this
sparse one (6.4%).

Caveat unchanged and now dominant: one seed per variant. The Poisson counting
error on the ratio is +-2.8% at 5 dB, but seed-to-seed training variance is not
measured and is almost certainly larger than the 5-15% effect being claimed.

Figure: `exp/fer-ldpc-rptu-96-48-trackA.png`.

## Infrastructure note (2026-09-18)

The `/Codes` NFS export degraded badly for several hours (reading one venv `.so`
took 2m47s vs 0.3s on local disk). Three A1 attempts died identically: all four
DDP ranks stuck in state D on `rpc_wait_bit_killable`, rank 0 needing ~40 min just
to create its Hydra run dir, workers missing Lightning's 1801 s rendezvous timeout
-> `DistStoreError: ... 1/4 clients joined`. Nothing wrong with the repo. If it
recurs: check `timeout 25 cat .venv/.../libtorch_cuda.so` before relaunching, and
keep run logs on local disk (4 ranks writing tqdm output to one NFS file is 20 MB
of carriage returns per run).

Also worth knowing: `sbnd-test` is single-GPU (`src/test.py` hardcodes `cuda`), so
evals of different checkpoints should be run concurrently with `CUDA_VISIBLE_DEVICES`
rather than sequentially — 4 h each, and they do not slow each other down.

## Phase 3 — BCH(63,45,7), DONE

`recct-bch-63-45-ml-4m-2dB-aug` (4M ML @ 2 dB + BCHPerms aug) with `max_epochs=128`
only; the config's own `warmup 10 / decay 32` kept, so again a complete compressed
WSD schedule. ~2h47/run. Eval 1.0-6.0 dB step 1.0, 33.55M cw/point.

| id | W&B run           | val/loss | val/acc | FER @4 dB | @5 dB    | @6 dB    |
|----|-------------------|----------|---------|-----------|----------|----------|
| A0 | golden-valley-2113 | 0.0648  | 0.686   | 1.772e-2  | 1.269e-3 | 3.380e-5 |
| A1 | decent-planet-2114 | 0.0649  | 0.687   | 1.605e-2  | 1.036e-3 | 2.235e-5 |

FER ratio A1/A0: 1.003 / 0.996 / 0.967 / 0.906 / 0.817 / 0.661 at 1-6 dB
(+-1 sigma 0.000 / 0.000 / 0.001 / 0.002 / 0.007 / 0.047 from counting alone).
The curve falls 1.57 decades/dB there, so 0.661 is ~0.11 dB.

A1 is level with or very slightly worse than A0 at 1-2 dB, i.e. near the 2 dB
training point, and pulls ahead only away from it. The same shape holds on all
three codes, but the SIZE of the margin is not explained by anything simple.
Compared at a common 5 dB:

| code        | n  | m  | H density | row wt | col wt | A1/A0 @5 dB |
|-------------|----|----|-----------|--------|--------|-------------|
| eBCH(32,16) | 32 | 16 | 0.297     | 8-12   | 1-11   | 0.958       |
| BCH(63,45)  | 63 | 18 | 0.325     | 16-28  | 1-11   | 0.817       |
| LDPC(96,48) | 96 | 48 | 0.064     | 6-7    | 3-4    | 0.852       |

Neither blocklength nor H density orders these: BCH(63,45) has the largest margin
while being shorter than the LDPC code, and eBCH and BCH have nearly the same
density and column-weight profile but the smallest and largest margins. Coset
weight distribution is the more likely place to look. Note also that the three
setups differ in training SNR (3 / 2 / 3 dB), data regime (ML file + augmentation
vs on-demand) and schedule, so cross-code margins are confounded beyond the code
itself. With one seed per point, three codes cannot decompose this.

Note the validation metrics do NOT show this: val/loss and val/acc are identical
to three digits (0.0649 vs 0.0648, 0.687 vs 0.686) and A1's final train/loss is
slightly WORSE (0.0692 vs 0.0678). Validation is measured at the 2 dB training
SNR, where the FER curves also coincide. Judging these variants on val metrics
would have missed the effect entirely.

Figure: `exp/fer-bch-63-45-trackA.png`.

## Known repo gotchas hit along the way

- `dev-test-*` presets set `offline: true`, which masks any W&B-dependent bug.
  Smoke tests of W&B behavior need an explicit `offline=false`.
- Overriding `max_epochs` below `warmup + decay` trips an assert in
  `sbnd.lr_sched.WarmupStableDecayLR`.
- Fixed on main (`8435f66`): `WandbModifyCheckpointName` read
  `logger.experiment.name` on every DDP rank, but only rank 0 has a real
  `wandb.Run`; the others got a `_DummyExperiment` bound method baked into the
  checkpoint filename, which broke `trainer.test(ckpt_path="best")` after fit.
