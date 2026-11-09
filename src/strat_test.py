# Weight-stratified Monte Carlo evaluation of an SBND model, for WERs far below the
# reach of plain Monte Carlo.
#
# WER(g) = sum_w P_w(g) F_w(g), with P_w the Binomial(n, p) pmf of the number w of
# hard-decision errors (p = Q(1/sigma)) and F_w the failure rate given exactly w hard
# errors, estimated on received words sampled conditionally on w. The words go through
# sbnd-test's input path (prepare_data), decoding (decode_batch, with the configured
# TTS) and error counting (update_error_stats, with HDD if enabled), so F_w is
# sbnd-test's WER restricted to weight w.

import os, sys, csv, math, pathlib
from typing import Any
import torch, hydra

from torch import Tensor
from torch.special import ndtr, ndtri
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig

from .utils import get_rank_zero_logger
from .codes import LinearCode
from .data import ebno_to_sigma, prepare_data
from .test import (
    COUNTS,
    PRECISIONS,
    _csv_lock,
    _snr_key,
    decode_batch,
    load_lit_model,
    setup_eval,
    setup_model,
    update_error_stats,
)

log = get_rank_zero_logger(__name__)

CELL_COLUMNS = ["Eb/N0", "w", "P_w", *COUNTS]
AGG_COLUMNS = ["Eb/N0", "WER", "BER", "ci95", "upper", "Total CW", "w_max"]
Z95 = 1.96


def hard_error_prob(sigma: float) -> float:
    """p = Q(1/sigma): probability that y_i = 1 + sigma z_i is negative."""
    return ndtr(torch.tensor(-1 / sigma, dtype=torch.float64)).item()


def binom_pmf(n: int, p: float) -> list[float]:
    """Binomial(n, p) pmf, computed in log domain in float64."""
    lp, lq = math.log(p), math.log1p(-p)
    lg = math.lgamma
    return [
        math.exp(lg(n + 1) - lg(w + 1) - lg(n - w + 1) + w * lp + (n - w) * lq)
        for w in range(n + 1)
    ]


def sample_w_errors(n: int, w: int, sigma: float, bs: int) -> tuple[Tensor, Tensor]:
    """Draw bs received words y = 1 + sigma z (all-zero codeword) with exactly w
    negative entries, at w positions drawn uniformly without replacement.

    z is a standard Gaussian truncated to (-inf, -a) on the error positions and to
    (-a, inf) elsewhere (a = 1/sigma), drawn by inverse CDF in float64 with ndtri only
    called on arguments <= 1/2 (its accurate side). Returns y (float64) and e = (y < 0).
    """
    a = 1 / sigma
    q = ndtr(torch.tensor(-a, dtype=torch.float64))  # P(Z < -a)
    # u in (0, 1]: avoids ndtri(0) = -inf
    # other side: P(Z > z) = v = u (1 - q), i.e. z = -ndtri(v) for v <= 1/2 and
    # z = ndtri(1 - v) above (1 - v is exact there, Sterbenz): one call on min(v, 1 - v)
    v = (1 - torch.rand(bs, n, dtype=torch.float64)) * (1 - q)
    z = ndtri(torch.minimum(v, 1 - v)) * torch.where(v <= 0.5, -1.0, 1.0)
    # error side, on the w drawn positions only: P(Z < z) = u q <= q
    pos = torch.rand(bs, n).topk(w, dim=1).indices
    u = 1 - torch.rand(bs, w, dtype=torch.float64)
    z.scatter_(1, pos, ndtri(u * q))
    err = torch.zeros(bs, n, dtype=torch.bool).scatter_(1, pos, True)
    y = 1 + sigma * z
    # z at the truncation point (u = 1, or rounding of 1 - sigma a) would put y on 0
    # or on the wrong side: keep the sign, even after the float32 cast of the model input
    tiny = torch.finfo(torch.float32).tiny
    y = torch.where(err, y.clamp(max=-tiny), y.clamp(min=0.0))
    return y, (y < 0).to(torch.int8)


class WeightConditionedDataset(Dataset):
    """n_batches batches of bs words with exactly w hard errors, prepared as in sbnd-test."""

    def __init__(
        self,
        code: LinearCode,
        sigma: float,
        w: int,
        n_batches: int,
        bs: int,
        error_space: str,
    ) -> None:
        self.code, self.sigma, self.w = code, sigma, w
        self.n_batches, self.bs, self.error_space = n_batches, bs, error_space

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        if not 0 <= idx < self.n_batches:
            raise IndexError(idx)
        y, e = sample_w_errors(self.code.n, self.w, self.sigma, self.bs)
        return prepare_data(self.code, y.float(), e, self.error_space)

    def __len__(self) -> int:
        return self.n_batches


Cells = dict[tuple[float, int], dict[str, float]]


def load_cells(path: str) -> Cells:
    if not pathlib.Path(path).exists():
        return {}
    with open(path, newline="") as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    for r in rows:
        r["w"] = int(r["w"])  # written as an integer, like a fresh cell's w
    return {(_snr_key(r["Eb/N0"]), int(r["w"])): r for r in rows}


def write_rows(rows: list[dict], columns: list[str], path: str) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def merge_cell(
    path: str, snr: float, w: int, P_w: float, delta: dict[str, float]
) -> Cells:
    """Add this run's counts for cell (snr, w) to the ones on disk; return all cells.
    Same locking as sbnd-test's merge_into_csv, so concurrent runs add up."""
    with _csv_lock(path):
        cells = load_cells(path)
        row = cells.setdefault((snr, w), {c: 0.0 for c in CELL_COLUMNS})
        row.update({"Eb/N0": snr, "w": w, "P_w": P_w})
        for c in COUNTS:
            row[c] += delta[c]
        write_rows([cells[k] for k in sorted(cells)], CELL_COLUMNS, path)
    return cells


def covered_weights(cells: Cells, snr: float) -> int:
    """Largest w_max such that weights 1..w_max all have a cell at this SNR."""
    w_max = 0
    while (snr, w_max + 1) in cells:
        w_max += 1
    return w_max


def aggregate(cells: Cells, snr: float, P: list[float], k: int) -> dict[str, Any]:
    """Combine the cells of one SNR into its WER and BER.

    Weights 1..w_max must be contiguous; the uncovered tail P(W > w_max) is added to
    the upper bound (F_w <= 1), as are the rule-of-three bounds of zero-failure cells.
    """
    w_max = covered_weights(cells, snr)
    tail = max(1.0 - sum(P[: w_max + 1]), 0.0)
    wer = var = zero = ber = frames = 0.0
    for w in range(1, w_max + 1):
        c = cells[(snr, w)]
        N, f = c["Total CW"], c["CW errors"]
        frames += N
        ber += P[w] * c["Bit errors"] / (N * k)
        if f > 0:
            F = f / N
            wer += P[w] * F
            var += P[w] ** 2 * F * (1 - F) / N
        else:  # rule of three: F_w < 3/N at 95 %
            zero += P[w] * 3 / N
    ci = Z95 * math.sqrt(var)
    return {
        "Eb/N0": snr,
        "WER": wer,
        "BER": ber,
        "ci95": ci,
        "upper": wer + ci + zero + tail,
        "Total CW": frames,
        "w_max": w_max,
    }


def cell_scores(
    cells: Cells, snr: float, P: list[float], ref: float
) -> dict[int, float]:
    """Relative variance contribution of each covered weight. A zero-failure cell
    counts its rule-of-three bound P_w 3/N as a 95 % half-width, i.e.
    (P_w 3 / N / 1.96)^2 as a variance."""
    scores = {}
    for w in range(1, covered_weights(cells, snr) + 1):
        c = cells[(snr, w)]
        N, f = c["Total CW"], c["CW errors"]
        F = f / N
        v = P[w] ** 2 * (F * (1 - F) / N if f > 0 else (3 / N / Z95) ** 2)
        scores[w] = v / ref**2
    return scores


def run_cell(
    model: torch.nn.Module,
    code: LinearCode,
    tts: object,
    sigma: float,
    w: int,
    frames: int,
    cfg: DictConfig,
    t: int,
    Ginv: Tensor,
    error_space: str,
) -> dict[str, float]:
    """Decode `frames` words with exactly w hard errors; return the counts."""
    n_batches = max(1, round(frames / cfg.batch_size))
    ds = WeightConditionedDataset(
        code, sigma, w, n_batches, cfg.batch_size, error_space
    )
    dl = DataLoader(ds, batch_size=None, num_workers=cfg.num_workers)
    delta = {c: 0.0 for c in COUNTS}
    device = Ginv.device
    with torch.no_grad():
        for ym, syndromes, targets in dl:
            synd = syndromes.to(device)
            tgt = targets.to(device)
            preds = decode_batch(
                model, code, tts, ym.to(device), synd, tgt.shape, cfg.precision  # type: ignore[arg-type]
            )
            update_error_stats(Ginv, error_space, preds, tgt, synd, delta, t)
    return delta


def run_snr(
    model: torch.nn.Module,
    code: LinearCode,
    tts: object,
    ebno_dB: float,
    cfg: DictConfig,
    t: int,
    Ginv: Tensor,
    error_space: str,
    cell_file: str,
) -> Cells:
    """Allocate the frames of one SNR point across weights, until the WER's relative
    95 % half-width is <= cfg.rel_ci, its upper bound < cfg.stop_below, or
    cfg.max_frames frames are on disk."""
    snr = _snr_key(ebno_dB)
    sigma = ebno_to_sigma(code.rate, ebno_dB)
    P = binom_pmf(code.n, hard_error_prob(sigma))
    cells = load_cells(cell_file)

    def extend(w: int, frames: int) -> None:
        nonlocal cells
        delta = run_cell(model, code, tts, sigma, w, frames, cfg, t, Ginv, error_space)
        cells = merge_cell(cell_file, snr, w, P[w], delta)
        c = cells[(snr, w)]
        print(
            f"  {ebno_dB:.2f} dB  w={w:2d}  +{int(delta['Total CW']):,}  "
            f"N={int(c['Total CW']):,}  errors={int(c['CW errors'])}  P_w={P[w]:.3e}"
        )

    # init_frames on each weight from 1 up to d_min (nonzero syndromes, so the model
    # always runs), then up to the first weight whose tail P(W > w) is below
    # tail_rel x the WER (or its upper bound while no failure is seen). Without a
    # known d_min, the tail rule alone decides from w = 1.
    w_lo = min(code.dmin or 1, code.n)
    for w in range(1, code.n + 1):
        have = cells.get((snr, w), {"Total CW": 0.0})["Total CW"]
        if have < cfg.init_frames:
            extend(w, int(cfg.init_frames - have))
        if w >= w_lo:
            a = aggregate(cells, snr, P, code.k)
            if 1.0 - sum(P[: w + 1]) < cfg.tail_rel * (a["WER"] or a["upper"]):
                break

    while True:
        a = aggregate(cells, snr, P, code.k)
        wer, upper = a["WER"], a["upper"]
        if (wer > 0 and upper - wer <= cfg.rel_ci * wer) or upper < cfg.stop_below:
            print(f"  {ebno_dB:.2f} dB: converged")
            break
        room = cfg.max_frames - int(a["Total CW"])
        if room < cfg.batch_size:
            print(f"  {ebno_dB:.2f} dB: frame cap reached ({int(a['Total CW']):,})")
            break
        scores = cell_scores(cells, snr, P, wer or upper)
        w = max(scores, key=lambda x: scores[x])
        N = int(cells[(snr, w)]["Total CW"])
        extend(w, min(N, cfg.max_chunk, room))  # doubling, in bounded chunks
    return cells


def write_agg(cells: Cells, code: LinearCode, path: str) -> list[dict]:
    rows = []
    for snr in sorted({s for (s, _) in cells}):
        P = binom_pmf(code.n, hard_error_prob(ebno_to_sigma(code.rate, snr)))
        rows.append(aggregate(cells, snr, P, code.k))
    write_rows(rows, AGG_COLUMNS, path)
    return rows


# conf/ is not part of the installed package; it lives in the project root.
_conf_dir = os.path.join(os.getcwd(), "conf")


@hydra.main(version_base="1.3", config_path=_conf_dir, config_name="strat_test")
def _main(cfg: DictConfig) -> None:
    if torch.cuda.is_available():
        torch.set_num_threads(1)
    if cfg.num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {cfg.num_workers}")
    if cfg.precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")

    model_file = cfg.model if cfg.model.endswith(".ckpt") else cfg.model + ".ckpt"
    log.info(f"Loading model from file: {model_file}")
    model = load_lit_model(model_file)
    error_space = getattr(model.decoder, "error_space", "codeword")
    code_file = model.hparams.code_path  # type: ignore[attr-defined]
    code = LinearCode(code_file if code_file.endswith(".mat") else code_file + ".mat")
    log.info(f"Code {code}, precision {cfg.precision}")
    tts, t, suffix = setup_eval(cfg, model, code)
    model, device = setup_model(model)  # type: ignore[assignment]
    Ginv = code.Ginv.float().to(device)

    pathlib.Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.output_dir}/{pathlib.Path(model_file).stem}{suffix}-strat"
    cell_file, agg_file = f"{stem}.csv", f"{stem}-agg.csv"
    log.info(f"Per-(SNR, w) counts: {cell_file}; aggregated WERs: {agg_file}")
    if cfg.seed is not None:
        # once: re-running with the same seed on an existing CSV repeats frames
        torch.manual_seed(cfg.seed)

    snrs = torch.arange(cfg.snr_min, cfg.snr_max + cfg.snr_step / 2, cfg.snr_step)
    for ebno_dB in snrs.tolist():
        print(f"Eb/N0 = {ebno_dB:.2f} dB")
        cells = run_snr(model, code, tts, ebno_dB, cfg, t, Ginv, error_space, cell_file)
        for r in write_agg(cells, code, agg_file):
            if r["Eb/N0"] == _snr_key(ebno_dB):
                print(
                    f"  WER {r['WER']:.3e} +- {r['ci95']:.1e}  upper {r['upper']:.3e}  "
                    f"BER {r['BER']:.3e}  ({int(r['Total CW']):,} frames, "
                    f"w_max {r['w_max']})"
                )


def main() -> None:
    """Console-script entry point; the first positional argument is the checkpoint,
    as for sbnd-test."""
    if len(sys.argv) >= 2:
        first = sys.argv[1]
        if not (first.startswith(("+", "~", "-")) or "=" in first):
            sys.argv[1] = f"model={first}"
    _main()


if __name__ == "__main__":
    main()
