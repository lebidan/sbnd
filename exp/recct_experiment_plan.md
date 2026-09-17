# rECCT: input embedding and iterative decoding. Experiment plan

Repository: https://github.com/lebidan/sbnd/ (module `recct.py`)
Status: draft for step-by-step exploration with Claude Code.

---

## 1. Problem statement

### 1.1 Context

Syndrome-based neural decoding (SBND) of linear block codes with a transformer
(ECCT, Choukroun & Wolf, arXiv:2203.14966). Our recurrent variant rECCT (looped /
universal-transformer style, one shared encoder block iterated `n_iters` times)
reaches near-ML FER on BCH(63,45) with 611K parameters, clearly ahead of the
original 1.2M-parameter ECCT.

### 1.2 The issue

Both ECCT and rECCT scale poorly to longer codes and codes with larger `dmin`.
One suspected contributor is the input embedding inherited from ECCT:

```
x = cat(|y|, s)                      # [B, n+m], |y| >= 0, s in {-1,+1}  (bipolar syndrome)
x_i = x_i * e_i                      # e_i: learned row i of an (n+m) x d table
```

Reliabilities and syndrome bits are information of a different nature, yet both
are used the same way: as amplitude modulation of a learned per-position vector.

### 1.3 Mechanistic diagnosis (why the multiplicative embedding is fragile)

The encoder block is pre-norm. For any scalar `c > 0`,
`LayerNorm(c * e_i) = LayerNorm(e_i)` (up to epsilon). Consequences:

1. The **first attention sub-layer sees no reliability at all**: only position
   (`e_i`) and syndrome sign (`s_j = +-1` survives normalization).
2. Reliability survives only in the residual stream, encoded by the *norm ratio*
   between `|y_i| e_i` and the attention / FFN updates accumulated on top of it.
3. This encoding **dilutes with every loop iteration**: `x_T = x_0 + sum_t delta_t`,
   the residual grows, `x_0` keeps a fixed norm, and pre-LN only sees direction.
   With 6 iterations and no re-injection of the input, the channel information is
   progressively washed out. The effect plausibly worsens with code length.

### 1.4 Inspiration

- Universal / looped transformers (Dehghani et al. 2019; Fan et al. 2024;
  Yang et al. 2024): weight sharing across depth, input re-injection at every loop,
  adaptive number of iterations.
- Classical iterative decoding (BP, min-sum, bit flipping, Chase): every iteration
  re-reads the channel LLRs, and a residual syndrome tells whether a valid codeword
  has been reached.
- DDECC (Choukroun & Wolf, ICLR 2023): iterative neural decoding driven by the
  syndrome of the current estimate.
- LLM input pipeline: token content embedding + positional embedding, added, not
  multiplied.

### 1.5 Hypotheses to test

- **H1**: an additive embedding (position + content) lets reliability information
  reach the first attention layer and improves FER and/or scaling.
- **H2**: re-injecting the input embedding at each loop iteration prevents dilution
  and makes extra iterations useful (also at inference).
- **H3**: turning the loop into a true iterative decoder whose state is the error
  estimate, with the residual syndrome as control signal, improves FER on hard codes
  and yields a natural stopping criterion and list-decoding extension.

Tracks A, B and C below map to H1, H2, H3. A and B are cheap and complementary;
C subsumes both (its first iteration *is* A + B).

---

## 2. Common experimental protocol

Apply identically to every track so results are comparable.

### 2.1 Codes

- Reference: BCH(63,45) (existing ML, Chase-2 and ECCT curves available).
- Short sanity code: BCH(31,16) (fast turnaround for screening).
- Scaling: RPTU LDPC (96,48,10) and CCSDS LDPC (128,64,14). Both have sparse `H`
  (relevant for the soft syndrome product of track C, and for the attention mask
  structure) and a `dmin` well above the BCH references.
- All codes, training configs and reference rECCT results are in the `sbnd` repo;
  reuse them verbatim.

### 2.2 Baseline

`RECCT` as in `recct.py` with the current default configuration
(`d=64, n_heads=4, n_layers=1, n_iters=6`). Re-run it under the same training
budget as every variant so that the comparison is fair (same seeds, same number of
optimizer steps, same batch size, same Eb/N0 training range, same LR schedule).

### 2.3 Parameter matching

Report the parameter count of every variant. When a variant adds parameters
(e.g. a second embedding table), also run a baseline with `d` increased to match,
so gains are not explained by capacity alone.

### 2.4 Metrics

- FER and BER vs Eb/N0 (2 to 6 dB, 0.5 dB step on BCH(63,45)), 95% confidence
  intervals, minimum 100 frame errors per point (fewer at low FER if budget limited).
- Gap to ML at FER = 1e-3 and 1e-5 (in dB).
- Training curves (loss and validation FER at a fixed Eb/N0) for early comparison.
- For tracks B and C: FER as a function of the number of iterations at inference
  (`n_iters` in {2, 4, 6, 8, 12, 16}), with a model trained at 6.
- For track C: syndrome satisfaction rate per iteration, and fraction of frames that
  stop early.

### 2.5 Bookkeeping

- Seeds: 3 per configuration at minimum for the final comparison; 1 for screening.
- Every run tagged with `track`, `variant`, `code`, `seed`; results appended to a
  single CSV / JSON log so plots can be regenerated.
- One config file per variant (no code branches selected by hand-edited constants).

### 2.6 Confirmed conventions and engineering constraints

- Inputs (see `prepare_data` in `src/data.py`): `s` is the **bipolar** syndrome in
  `{-1, +1}`; `ym` is the bit reliability `|y_m|` **normalized to [0, 1] by the max
  reliability of the current codeword**. Consequences:
  - scalar features operate on `[0, 1]`: `log1p` brings little, Fourier features
    with frequencies in `[1, 16]` cover the range; `bins` use uniform edges on `[0, 1]`;
  - the absolute LLR scale is lost, so any "analog weight" guard in track C ranks
    candidates within a frame only, which is sufficient for list decoding.
- Loss lives in `SBNDLitModule` (`src/model.py`). Track C returns one logits
  tensor per iteration; `forward` keeps returning the **last** logits by default
  (evaluation and compile paths unchanged) and exposes `forward_all` used by
  `training_step` for deep supervision.
- Training budget: the same config as the reference rECCT runs of the repo, per
  code. No tuning of optimizer, schedule or Eb/N0 range unless stated.
- **`compile=True` and bf16-mixed are the reference from day one.** Rules:
  - the number of iterations `T` is a Python int fixed per run (loop unrolled at
    trace time); variants with several `T` values (B5, curriculum) use a small
    fixed set (e.g. `{4, 6, 8}`) and accept one compilation per value;
  - no data-dependent control flow inside the compiled region: early stopping in
    track C is done by compiling the per-iteration step and keeping the loop and
    the `done` mask in eager mode for evaluation;
  - the soft syndrome product of track C is computed in **fp32** under
    `torch.autocast(enabled=False)`: a product of many `(-1, 1)` values in bf16
    underflows and loses sign information;
  - every new module is checked once with `torch.compile(fullgraph=True)` to catch
    graph breaks early.

---

## 3. Track A: additive embedding (cheap, several options)

Goal: replace `x_i = value_i * e_i` by `x_i = position_i + content_i`.
Only `EmbeddingLayer` changes. Same encoder, same mask, same head.

Note: a first attempt at A1 was reportedly inconclusive. Re-run it cleanly under
the protocol of section 2 before drawing conclusions, then move to A2 which is the
more principled design.

### A1. Affine (two tables), minimal change

Both bit and syndrome tokens get the same treatment. For syndrome tokens with
`s_j = +-1` this yields `p_{n+j} +- u_{n+j}`, i.e. a proper two-symbol embedding
with `p_{n+j}` carrying the identity of the check.

```python
class AffineEmbedding(nn.Module):
    def __init__(self, n_tokens: int, d: int) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.randn(n_tokens, d) * 0.02)  # identity
        self.val = nn.Parameter(torch.randn(n_tokens, d) * 0.02)  # content direction

    def forward(self, ym: Tensor, s: Tensor) -> Tensor:
        x = torch.cat([ym, s], dim=1)                     # [B, n+m]
        return self.pos[None] + x[..., None] * self.val[None]   # [B, n+m, d]
```

Variant A1b (control): keep the original multiplicative syndrome tokens
`s_j * e_{n+j}` and apply the affine form to the `n` bit tokens only. This isolates
the reliability part of the change.

### A2. Decoupled content: shared reliability MLP + two-symbol syndrome table

Position is a per-token table (the code is fixed, so per-position parameters are
legitimate). Reliability content is produced by a small MLP **shared across
positions**, from scalar features. Syndrome content is a `2 x d` table indexed by the
binary syndrome bit.

```python
def reliability_features(ym: Tensor, kind: str) -> Tensor:
    # ym: [B, n] -> [B, n, F]
    if kind == "raw":
        return ym[..., None]                                   # F = 1
    if kind == "raw_log":
        return torch.stack([ym, torch.log1p(ym)], dim=-1)      # F = 2
    if kind == "fourier":
        # fixed log-spaced frequencies, like diffusion timestep embeddings
        w = torch.logspace(0, 4, steps=8, base=2, device=ym.device)  # 1 .. 16, ym in [0, 1]
        arg = ym[..., None] * w                                # [B, n, 8]
        return torch.cat([ym[..., None], arg.sin(), arg.cos()], dim=-1)  # F = 17
    if kind == "bins":
        # handled by a separate nn.Embedding, see A2c
        raise NotImplementedError


class DecoupledEmbedding(nn.Module):
    def __init__(self, n: int, m: int, d: int, feat: str = "raw_log", hidden: int | None = None) -> None:
        super().__init__()
        self.n, self.m = n, m
        F = {"raw": 1, "raw_log": 2, "fourier": 17}[feat]
        self.feat = feat
        self.pos = nn.Embedding(n + m, d)
        hidden = hidden or d
        self.rel = nn.Sequential(nn.Linear(F, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.syn = nn.Embedding(2, d)      # symbols 0 / 1

    def forward(self, ym: Tensor, s: Tensor) -> Tensor:
        # ym: [B, n] magnitudes; s: [B, m] bipolar (+1 -> syndrome bit 0, -1 -> bit 1)
        s_bin = (s < 0).long()
        xb = self.pos.weight[: self.n][None] + self.rel(reliability_features(ym, self.feat))  # [B, n, d]
        xs = self.pos.weight[self.n :][None] + self.syn(s_bin)                                 # [B, m, d]
        return torch.cat([xb, xs], dim=1)
```

Sub-variants (screen with 1 seed, keep the best):

- **A2a** `feat="raw"`: strictly minimal content, tests the additive form alone.
- **A2b** `feat="raw_log"` or `"fourier"`: more expressive scalar encoding.
- **A2c** binned reliability: quantize `|y|` into `K` bins (e.g. K = 16, quantile
  edges estimated on training data) and use `nn.Embedding(K, d)` instead of the MLP.
  Expected to lose resolution; useful as a sanity check of what matters.
- **A2d** per-position reliability direction: `content_i = phi(|y_i|) * v_i` with
  `phi` a shared scalar MLP `R -> R` and `v_i` a per-position vector (mid-way between
  A1 and A2).

### A3. Fold the local syndrome into the bit tokens (sequence length n)

Removes the `m` syndrome tokens. Each bit token receives the signed contribution of
the checks it participates in. Attention mask reduces to the `n x n` var-to-var
block. Reduces attention cost from `(n+m)^2` to `n^2`, relevant for long codes.

```python
class FoldedSyndromeEmbedding(nn.Module):
    def __init__(self, code, d: int, feat: str = "raw_log") -> None:
        super().__init__()
        n, m = code.n, code.m
        self.register_buffer("H", code.H.float())          # [m, n]
        self.pos = nn.Embedding(n, d)
        self.rel = ...                                     # as in A2
        self.check = nn.Parameter(torch.randn(m, d) * 0.02)  # one vector per check
        self.feat = feat

    def forward(self, ym: Tensor, s: Tensor) -> Tensor:
        # s bipolar [B, m]; local signed syndrome for bit i: {s_j : j in N(i)}
        loc = s[:, :, None] * self.H[None]                 # [B, m, n], zeros where H = 0
        syn_content = torch.einsum("bmn,md->bnd", loc, self.check)   # [B, n, d]
        return self.pos.weight[None] + self.rel(reliability_features(ym, self.feat)) + syn_content
```

Required changes: `register_mask` keeps only the top-left `n x n` block,
`DecoderLayer(fc_in=n)`. Expected trade-off: cheaper and shorter context, but the
transformer can no longer exchange messages through explicit check tokens.

### Track A run list

| id  | embedding                     | tokens | extra params | purpose                         |
|-----|-------------------------------|--------|--------------|---------------------------------|
| A0  | baseline multiplicative       | n+m    | 0            | reference                       |
| A1  | affine, both blocks           | n+m    | (n+m) d      | minimal additive test           |
| A1b | affine bits only              | n+m    | n d          | isolate reliability part        |
| A2a | decoupled, raw                | n+m    | ~2 d^2       | principled additive             |
| A2b | decoupled, raw_log / fourier  | n+m    | ~2 d^2       | scalar encoding expressiveness  |
| A2c | decoupled, bins               | n+m    | K d          | sanity check                    |
| A3  | folded syndrome               | n      | m d + 2 d^2  | shorter context, scaling        |

Success criterion for H1: A2 (best sub-variant) beats A0 at equal budget on
BCH(63,45), and the gain grows on the longer / higher-`dmin` code.

---

## 4. Track B: re-inject the input embedding at every iteration

Goal: `x_{t+1} = block(x_t + x_0)` instead of `x_{t+1} = block(x_t)`.
Only `RECCT.forward` changes. Zero or near-zero extra parameters.

Rationale: keeps the channel information at constant weight in the residual stream
(no dilution), mimics iterative decoders that re-read the LLRs each iteration, and
makes the loop's fixed point depend on the input, which is what allows more
iterations at inference than at training.

Important: combine B with an additive embedding (track A). With the multiplicative
embedding, `x_0` re-injected still loses its scale through the pre-LN, so B alone
mostly restores position and syndrome sign.

### B1. Plain re-injection

```python
def forward(self, ym: Tensor, s: Tensor) -> Tensor:
    x0 = self.embed(ym, s)
    x = x0
    for layer in self.encoding_layers:
        for _ in range(self.n_iters):
            x = layer(x + x0, self.mask)
    return self.decode(x)
```

### B2. Learned scalar gate on the re-injection

```python
self.alpha = nn.Parameter(torch.ones(()))       # init 1.0; log its final value
...
x = layer(x + self.alpha * x0, self.mask)
```

Also try `alpha` initialized at 0 to check whether the model learns to use it.

### B3. Iteration embedding (universal-transformer style)

Lets the shared block behave differently at iteration 0 (raw input) and later
iterations (refinement), with negligible parameters.

```python
self.iter_emb = nn.Parameter(torch.zeros(max_iters, 1, 1, d))
...
for t in range(self.n_iters):
    x = layer(x + x0 + self.iter_emb[t], self.mask)
```

### B4. Concatenation + projection (stronger coupling)

```python
self.inject = nn.Linear(2 * d, d, bias=False)   # d^2 * 2 extra params
...
x = layer(self.inject(torch.cat([x, x0], dim=-1)), self.mask)
```

### B5. Iteration-count robustness during training

Sample `n_iters` uniformly in `[T_min, T_max]` per batch (e.g. 3 to 8). Evaluate at
several iteration counts. Cheap, and directly tests whether "more iterations at
inference" helps once re-injection is in place.

### Track B run list

| id | change                     | extra params | on top of         |
|----|----------------------------|--------------|-------------------|
| B1 | x + x0                     | 0            | A0 and best A     |
| B2 | x + alpha x0               | 1            | best A            |
| B3 | + iteration embedding      | T d          | best A + B1/B2    |
| B4 | concat + linear            | 2 d^2        | best A            |
| B5 | random T during training   | 0            | best A + best B   |

Success criterion for H2: FER improves or stays flat when `n_iters` at inference
exceeds the training value (baseline is expected to degrade); B1/B2 beats A-only
at equal budget.

---

## 5. Track C: iterative decoder on the residual syndrome (DDECC-flavoured refit)

Goal: make the loop a genuine iterative decoder. State = soft error estimate
`e_t in [0,1]^n`. At every iteration the tokens are rebuilt from
`(|y|, e_t, residual syndrome)`, the shared block predicts new logits, and the state
is updated. Iteration 0 (`e_0 = 0`) coincides with track A2 + B1.

### 5.1 Core design

**State and derived quantities**

```
e_t        in [0,1]^n         soft error estimate, e_0 = 0
u_t        = 1 - 2 e_t        in [-1,1]^n
sigma_t,j  = s_j * prod_{i in N(j)} u_t,i         residual syndrome, bipolar, soft, differentiable
y_corr_t,i = |y_i| * u_t,i                         corrected signed channel value
```

`sigma_t` is the sign product of min-sum applied to soft values; when `e_t` is a
hard pattern it equals `1 - 2 (s xor H e_t)`.

**Residual syndrome, two implementations**

```python
@torch.autocast("cuda", enabled=False)          # fp32 product, see section 2.6
def residual_syndrome(self, s: Tensor, e: Tensor, mode: str = "prod") -> Tensor:
    # s: [B, m] bipolar; e: [B, n] in [0,1]  ->  sigma: [B, m] in [-1, 1]
    s, e = s.float(), e.float()
    u = 1.0 - 2.0 * e                                        # [B, n]
    if mode == "prod":
        Hb = self.H.bool()                                   # [m, n]
        uu = torch.where(Hb[None], u[:, None, :], torch.ones_like(u)[:, None, :])  # [B, m, n]
        return s * uu.prod(dim=-1)
    if mode == "hard_ste":
        # hard sign product in the forward pass, straight-through gradient of u
        u_hard = torch.sign(u).detach() + (u - u.detach())
        Hb = self.H.bool()
        uu = torch.where(Hb[None], u_hard[:, None, :], torch.ones_like(u)[:, None, :])
        return s * uu.prod(dim=-1)
```

Caveat for dense `H` (BCH rows have many ones): a product of many soft values in
`(-1,1)` collapses toward 0, killing both signal and gradient. Mitigations to test:
`hard_ste`; sharpen `e_t` with a temperature before the product; or compute
`prod` in the log domain with a magnitude floor. Log the mean `|sigma_t|` per
iteration to detect collapse.

**Token construction (rebuilt every iteration)**

```python
class StateEmbedding(nn.Module):
    def __init__(self, n: int, m: int, d: int, feat: str = "raw_log", state_mode: str = "corrected") -> None:
        super().__init__()
        self.n, self.m, self.feat, self.state_mode = n, m, feat, state_mode
        self.pos = nn.Embedding(n + m, d)
        F = {"raw": 1, "raw_log": 2, "fourier": 17}[feat]
        # signed features: reuse reliability_features on |y_corr| plus the sign as an extra feature
        self.rel = nn.Sequential(nn.Linear(F + 1, d), nn.GELU(), nn.Linear(d, d))
        self.state = nn.Parameter(torch.randn(2, d) * 0.02)   # v0, v1 for e = 0 / 1 (state_mode="symbols")
        self.check = nn.Parameter(torch.randn(m, d) * 0.02)   # u_j direction for sigma_j

    def forward(self, ym: Tensor, e: Tensor, sigma: Tensor) -> Tensor:
        u = 1.0 - 2.0 * e                                                  # [B, n]
        if self.state_mode == "corrected":
            f = torch.cat([reliability_features(ym, self.feat), u[..., None]], dim=-1)
            content = self.rel(f)                                           # sign-aware content
        else:  # "symbols": reliability content + soft two-symbol state
            f = torch.cat([reliability_features(ym, self.feat), torch.zeros_like(u)[..., None]], dim=-1)
            content = self.rel(f) + e[..., None] * self.state[1] + (1 - e)[..., None] * self.state[0]
        xb = self.pos.weight[: self.n][None] + content                      # [B, n, d]
        xs = self.pos.weight[self.n :][None] + sigma[..., None] * self.check[None]   # [B, m, d]
        return torch.cat([xb, xs], dim=1)
```

**Forward with state update, deep supervision outputs**

```python
class IterativeRECCT(BaseDecoder):
    def __init__(self, code, d=64, n_heads=4, n_layers=1, n_iters=6,
                 update="replace", detach_state=False, syn_mode="prod", tau=1.0, **kw):
        super().__init__(code, **kw)
        self.n_iters, self.update, self.detach_state, self.syn_mode, self.tau = n_iters, update, detach_state, syn_mode, tau
        self.register_mask(code)
        self.register_buffer("H", code.H.float())
        self.embed = StateEmbedding(code.n, code.m, d)
        self.block = nn.ModuleList([EncoderLayer(d, n_heads, ...) for _ in range(n_layers)])
        self.head = DecoderLayer(d, code.n + code.m, code.n)

    def forward(self, ym: Tensor, s: Tensor) -> Tensor:
        # default entry point: unchanged interface for eval / compile / SBNDLitModule.predict
        return self.forward_all(ym, s)[-1]

    def forward_all(self, ym: Tensor, s: Tensor, n_iters: int | None = None) -> list[Tensor]:
        # used by SBNDLitModule.training_step for deep supervision; T is a Python int
        T = n_iters or self.n_iters
        B = ym.shape[0]
        e = ym.new_zeros(B, self.code.n)          # e_0 = 0
        logits_all = []
        for t in range(T):
            e_in = e.detach() if self.detach_state else e
            sigma = self.residual_syndrome(s, e_in, self.syn_mode)
            x = self.embed(ym, e_in, sigma)
            for layer in self.block:
                x = layer(x, self.mask)
            delta = self.head(x)                  # [B, n] logits
            if self.update == "replace":
                logits = delta                    # block predicts the full error pattern
            elif self.update == "correct":
                p_flip = torch.sigmoid(delta)     # block predicts which bits to flip vs. current state
                e_new = e_in + p_flip - 2.0 * e_in * p_flip        # soft XOR
                logits = torch.logit(e_new.clamp(1e-6, 1 - 1e-6))
            logits_all.append(logits)
            e = torch.sigmoid(logits / self.tau)
        return logits_all                          # one [B, n] tensor per iteration
```

**Loss: deep supervision**

```python
def iterative_loss(logits_all: list[Tensor], e_true: Tensor, schedule: str = "linear") -> Tensor:
    T = len(logits_all)
    if schedule == "linear":
        w = torch.arange(1, T + 1, dtype=torch.float) / T
    elif schedule == "last":
        w = torch.zeros(T); w[-1] = 1.0
    elif schedule == "uniform":
        w = torch.ones(T) / T
    losses = [F.binary_cross_entropy_with_logits(l, e_true) for l in logits_all]
    return sum(wi * li for wi, li in zip(w, losses)) / w.sum()
```

Training with `schedule="last"` is expected to fail or to make early iterations
idle; it is a control run, not a candidate.

**Inference**

```python
def _step(self, ym, s, e):
    # one iteration, no control flow: this is the function to torch.compile
    sigma = self.residual_syndrome(s, e, "hard_ste")            # hard product at inference
    x = self.embed(ym, e, sigma)
    for layer in self.block:
        x = layer(x, self.mask)
    delta = self.head(x)
    return delta if self.update == "replace" else self._correct(e, delta)   # as in forward_all

@torch.no_grad()
def decode(self, ym, s, max_iters, early_stop=True):
    # eager loop around the compiled step; `done` mask handles per-frame early stopping
    e = ym.new_zeros(ym.shape[0], self.code.n)
    done = torch.zeros(ym.shape[0], dtype=torch.bool, device=ym.device)
    step = self._compiled_step or self._step
    for t in range(max_iters):
        logits = step(ym, s, e)
        e_new = torch.sigmoid(logits)
        e = torch.where(done[:, None], e, e_new)                # frozen frames keep their estimate
        if early_stop:
            e_hard = (e > 0.5).float()
            syn_ok = ((self.H @ e_hard.T).T % 2 == (s < 0).float()).all(dim=1)
            done |= syn_ok
            if done.all():
                break
    return (e > 0.5)
```

### 5.2 Core runs

| id  | update  | state_mode | syn_mode | detach | loss    | purpose                       |
|-----|---------|------------|----------|--------|---------|-------------------------------|
| C0  | replace | corrected  | prod     | no     | linear  | core design                   |
| C1  | replace | symbols    | prod     | no     | linear  | explicit 0/1 state embedding  |
| C2  | correct | corrected  | prod     | no     | linear  | flip-based update             |
| C3  | replace | corrected  | hard_ste | no     | linear  | dense-H robustness            |
| C4  | replace | corrected  | prod     | yes    | linear  | truncated BPTT through state  |
| C5  | replace | corrected  | prod     | no     | last    | control (no deep supervision) |

First sanity check for C0: with `n_iters=1` it must match A2 + B1 performance.
If it does not, the embedding or head differs from track A and must be aligned.

### 5.3 Refinements (after the core works)

- **Curriculum on T**: train at T = 2, then 4, then 6, warm-starting each stage.
- **Random T per batch** (as B5), evaluate at T up to 2x the training maximum.
- **Temperature annealing**: `tau` from 2.0 to 0.5 over training so that `e_t`
  becomes near-binary, closing the soft/hard gap with inference.
- **Weight-based guard at inference**: among frames whose syndrome is satisfied,
  reject an estimate whose analog weight `sum_i |y_i| e_i` is implausible (e.g.
  above a threshold calibrated on training data). Diagnoses convergence to wrong
  codewords.
- **Optional regularizer**: add `lambda * mean_t mean_i(|y_i| e_t,i)` to the loss to
  discourage heavy patterns (small `lambda`, check it does not hurt FER).
- **Iteration embedding** (B3) inside the loop.
- **List decoding** at inference: at iteration `t*`, branch on the two hypotheses of
  the least reliable undecided bit, run both to completion, keep the syndrome-valid
  candidate with minimum analog weight. Start with list size 2, then 4. This targets
  the large-`dmin` regime where one pass is insufficient.
- **Compile**: once T is fixed for evaluation, `torch.compile` the per-iteration
  step function only.

### 5.4 Diagnostics specific to track C

- Per-iteration FER and BER on a fixed validation set (should decrease monotonically).
- Mean `|sigma_t|` per iteration (detect collapse of the soft product).
- Histogram of the stopping iteration at inference; fraction of frames with
  satisfied syndrome but wrong codeword (undetected errors), by Eb/N0.
- Gradient norm per iteration during training (detects exploding BPTT; motivates C4).

---

## 6. Suggested order of execution

1. Confirm the reference rECCT results of the repo reproduce (A0) on BCH(31,16) and
   BCH(63,45) with the stored configs, compile=True, bf16-mixed. Record everything.
2. Screening on BCH(31,16) (fastest): A1, A1b, A2a (1 seed each). If A2a >= A1,
   drop A1 and proceed with A2.
3. A2b, A2c on BCH(31,16), confirm the best A on BCH(63,45).
4. B1 and B2 on top of best A; B1 on top of A0 as a control. Iteration sweep at inference.
5. B5 on best A + best B. Iteration sweep again.
6. First scaling test: best A + best B vs A0 on LDPC(96,48) and LDPC(128,64). This
   decides how much of the scaling problem was embedding / dilution.
7. A3 on the two LDPC codes only (its benefit is cost, not accuracy, on short codes;
   sparse `H` makes the folded syndrome cheap).
8. Track C core C0 on BCH(31,16): sanity check at T = 1 (must match A2 + B1), then
   T = 6, then C1 to C5 (1 seed). Sparse-`H` LDPC codes are where the soft syndrome
   product is best behaved; dense-`H` BCH codes are where C3 (`hard_ste`) matters.
9. Refinements of 5.3 on the best C, then 3 seeds on all four codes for the final
   comparison.
10. Final figures: FER vs Eb/N0 with A0, best A+B, best C, ECCT, Chase-2 / BP, ML
    where available, one per code.

---

## 7. Remaining open questions

1. Per-codeword max normalization of `ym`: worth an ablation with a global
   (per-Eb/N0 or fixed) normalization for tracks A2 and C, since the per-frame
   normalization erases the absolute noise level. Low priority, only if A2 gains
   are smaller than expected.
2. Compile with several `T` values (B5, curriculum): if recompilation cost is
   prohibitive, restrict to a single `T` for training and sweep `T` only at inference.

Resolved: the LDPC parity-check matrices are full rank, and the training / eval
code handles redundant `H` transparently, so the mask, the folded syndrome (A3)
and the residual syndrome (track C) are built from `code.H` as loaded, with
`m = H.shape[0]` whatever its rank.
