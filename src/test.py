# Evaluate the test performance of a trained SBND model through Monte Carlo simulations.

import os, sys, csv, fcntl, pathlib
import torch, hydra

from hydra.utils import instantiate
from torch import Tensor
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from contextlib import contextmanager
from typing import Generator
from tqdm import tqdm  # type: ignore[import-untyped]
from tabulate import tabulate  # type: ignore[import-untyped]

from .utils import get_rank_zero_logger
from .codes import LinearCode
from .model import SBNDLitModule
from .data import OnDemandDataset
from .tts import SingleShotDecoder

log = get_rank_zero_logger(__name__)


def load_lit_model(model_file: str) -> SBNDLitModule:
    return SBNDLitModule.load_from_checkpoint(model_file, weights_only=False)


def count_parameters(model: SBNDLitModule) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def bipolar_to_bit(x: Tensor) -> Tensor:
    return (x < 0).to(torch.int8)


# column labels for the output csv file
COLUMNS = ["Eb/N0", "WER", "BER", "CW errors", "Bit errors", "Total CW"]
# accepted `precision` values (Lightning naming)
PRECISIONS = ("32-true", "bf16-mixed")
# raw counters accumulated across runs (WER/BER are derived from them)
COUNTS = ("CW errors", "Bit errors", "Total CW")

# Decimal precision used when keying rows by Eb/N0. Picked large enough to
# distinguish the smallest SNR step we'd realistically use (0.01 dB), and
# small enough to absorb fp drift between torch.arange and CSV reading.
SNR_KEY_DECIMALS = 4


def _snr_key(x: float) -> float:
    """Round an Eb/N0 value to a stable precision for use as a dict key."""
    return round(x, SNR_KEY_DECIMALS)


def load_csv(path: str) -> list[dict[str, float]]:
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def write_csv(rows: list[dict[str, float]], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def _csv_lock(path: str) -> Generator[None, None, None]:
    # Lock a sidecar file, not the CSV: merge_into_csv replaces the CSV on each
    # write, voiding any lock on it. flock may be node-local on cluster filesystems.
    with open(path + ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield  # released when f is closed


def merge_into_csv(
    path: str, snr: float, delta: dict[str, float], k: int
) -> dict[float, dict[str, float]]:
    """Add this run's counts at `snr` to the counts already on disk; return all rows.

    Re-reading under the lock lets concurrent sbnd-test runs writing the same
    file (same or different SNR points) add up instead of overwriting each other.
    """
    with _csv_lock(path):
        rows = {}
        if pathlib.Path(path).exists():
            rows = {_snr_key(r["Eb/N0"]): r for r in load_csv(path)}
        row = rows.setdefault(snr, {c: 0.0 for c in COLUMNS})
        row["Eb/N0"] = snr
        for c in COUNTS:
            row[c] += delta[c]
        row["WER"] = row["CW errors"] / row["Total CW"]
        row["BER"] = row["Bit errors"] / (row["Total CW"] * k)
        tmp = path + ".tmp"  # safe to share: only written under the lock
        write_csv([rows[s] for s in sorted(rows)], tmp)
        os.replace(tmp, path)  # atomic: a crash mid-write never truncates the CSV
    return rows


def update_error_stats(
    Ginv: Tensor,
    error_space: str,
    preds: Tensor,
    targets: Tensor,
    syndromes: Tensor,
    stats: dict[str, float],
    t: int = 0,
) -> None:
    """
    Accumulate codeword (frame) and bit error counts over a test batch.

    BER is always reported on the k message bits. CW errors are reported on the
    full n-bit codeword when `error_space == "codeword"` (true FER), and on the
    k message bits otherwise (we don't have access to codeword-level errors when
    working in message mode). `Ginv` is `code.Ginv` as float, on the device of
    the other tensors.
    """
    # bit-level diff in the space where the model was trained (shape (bs, n) or (bs, k))
    # and in the message space (always (bs, k)) for BER counting
    diff_fer = preds != targets
    # Keep outside bf16 autocast: this GF(2) product is exact in fp32/TF32, bf16 only up to 256
    diff_ber = (diff_fer.float() @ Ginv) % 2 if error_space == "codeword" else diff_fer
    # all-zero error patterns don't even enter the decoder; nonzero ones with a zero
    # syndrome (+1 in bipolar form) are necessarily decoding errors
    counted = torch.any(targets != 0, dim=1)
    # emulate HDD: a detected error with at most t bit errors is a decoding success
    if t > 0:
        missed = torch.all(syndromes > 0, dim=1) | (diff_fer.sum(dim=1) > t)
        counted &= missed
    cw = (diff_fer.any(dim=1) & counted).sum()
    bits = (diff_ber.sum(dim=1) * counted).sum()
    cw, bits = torch.stack([cw, bits]).tolist()  # one GPU sync, no boolean indexing
    stats["Total CW"] += targets.size(0)
    stats["CW errors"] += cw
    stats["Bit errors"] += bits


def resolve_hdd_t(model: SBNDLitModule, code: LinearCode, hdd: bool) -> int:
    """Compute the HDD correction capability t from the code's dmin (0 if HDD is disabled)."""
    if not hdd:
        return 0
    error_space = getattr(model.decoder, "error_space", "codeword")
    if error_space == "message":
        raise ValueError(
            "hdd=true is only supported for models trained with error_space=codeword"
        )
    if code.dmin is None:
        raise ValueError(
            "hdd=true requires a code with a known dmin (not found in .mat file)"
        )
    return (code.dmin - 1) // 2


def test_model(
    code: LinearCode,
    model: SBNDLitModule,
    ebno_dB_range: Tensor,
    output_file: str,
    tts: object | None = None,
    test_bs: int = 4096,
    n_test_batches: int = 512,
    num_workers: int = 16,
    show_progress: bool = True,
    t: int = 0,
    min_cw_errors: int = 0,
    precision: str = "32-true",
) -> list[dict[str, float]]:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if tts is None:
        tts = SingleShotDecoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    model = model.to(device)
    model.eval()
    # After any model summary / FLOP count (sbnd-test runs none); no-op if compile=False
    if hasattr(model.decoder, "_maybe_compile"):
        model.decoder._maybe_compile()  # type: ignore[operator]
    error_space = getattr(model.decoder, "error_space", "codeword")
    Ginv = code.Ginv.float().to(device)  # no int8 matmul on CUDA
    # Each run only counts its own samples; merge_into_csv adds them to what is
    # on disk after each SNR point, so re-runs and concurrent runs accumulate.
    if pathlib.Path(output_file).exists():
        log.info(
            f"Appending to existing file: {output_file} ({len(load_csv(output_file))} rows already present)"
        )
    rows: dict[float, dict[str, float]] = {}
    # Setup and run MC simulation
    for ebno_dB in ebno_dB_range:
        snr = _snr_key(ebno_dB.item())
        print(f"Simulating Eb/N0 = {ebno_dB} dB")
        delta = {c: 0.0 for c in COUNTS}
        ds = OnDemandDataset(
            code,
            ebno_dB=ebno_dB,
            n_batches=n_test_batches,
            bs=test_bs,
            train=False,
            error_space=error_space,
        )
        dl = DataLoader(ds, batch_size=None, num_workers=num_workers)
        with torch.no_grad():
            for b, batch in enumerate(tqdm(dl, disable=not show_progress), 1):
                ym, syndromes, targets, _ = batch  # per-sample loss weight unused
                ym_dev = ym.to(device)
                synd_dev = syndromes.to(device)
                # Zero-syndrome words skip the model and stay uncorrected
                # (update_error_stats counts undetectable errors as errors anyway)
                nz = torch.any(synd_dev < 0, dim=1)
                preds = torch.zeros(targets.shape, dtype=torch.int8, device=device)
                if nz.any():
                    ym_nz, synd_nz = ym_dev[nz], synd_dev[nz]
                    # The batch size now varies: compile one dynamic-shape graph up front
                    torch._dynamo.maybe_mark_dynamic(ym_nz, 0)
                    torch._dynamo.maybe_mark_dynamic(synd_nz, 0)
                    with torch.autocast(
                        device.type, torch.bfloat16, enabled=precision == "bf16-mixed"
                    ):
                        preds[nz] = tts.decode(model, code, ym_nz, synd_nz)  # type: ignore[attr-defined]
                update_error_stats(
                    Ginv, error_space, preds, targets.to(device), synd_dev, delta, t
                )
                # This run's errors only: re-runs and concurrent runs each add >= min_cw_errors
                if min_cw_errors and delta["CW errors"] >= min_cw_errors:
                    print(
                        f"Stopped after {b}/{n_test_batches} batches ({int(delta['CW errors'])} CW errors)"
                    )
                    break
        rows = merge_into_csv(output_file, snr, delta, code.k)
        # print the cumulative stats at this SNR point
        print(rows[snr])
    if not rows and pathlib.Path(output_file).exists():
        return load_csv(output_file)
    return [rows[s] for s in sorted(rows)]


# conf/ is not part of the installed package; it lives in the project root.
# sbnd-test must be run from the directory that contains conf/.
_conf_dir = os.path.join(os.getcwd(), "conf")


@hydra.main(version_base="1.3", config_path=_conf_dir, config_name="test")
def _main(cfg: DictConfig) -> None:
    # CPU work here is only per-batch stats; default threads spin at 700%+ while GPU-bound
    torch.set_num_threads(1)
    if cfg.num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {cfg.num_workers}")

    # Load model first (code path is stored in its hparams)
    model_file = cfg.model
    if not model_file.endswith(".ckpt"):
        model_file += ".ckpt"
    log.info(f"Loading model from file: {model_file}")
    model = load_lit_model(model_file)
    log.info(f"Model {model} has been successfully loaded")
    log.info(f"This model has {count_parameters(model):,} trainable parameters")
    error_space = getattr(model.decoder, "error_space", "codeword")
    log.info(f"Model was trained with error_space={error_space}")

    # Resolve code file from the path stored in the checkpoint
    code_file = model.hparams.code_path  # type: ignore[attr-defined]
    if not code_file:
        raise ValueError("No code path found in checkpoint hparams.")
    if not code_file.endswith(".mat"):
        code_file += ".mat"
    log.info(f"Loading code from file: {code_file}")
    code = LinearCode(code_file)
    log.info(f"Code {code} has been successfully loaded")

    # Build the Eb/N0 sweep
    ebno_dB_range = torch.arange(cfg.snr_min, cfg.snr_max + cfg.snr_step, cfg.snr_step)
    log.info(
        f"Eb/N0 range to simulate: from {ebno_dB_range[0]} to {ebno_dB_range[-1]} by step of {cfg.snr_step} dB ({len(ebno_dB_range)} values)"
    )
    budget = f"{cfg.num_batches * cfg.batch_size:,} samples per Eb/N0 value ({cfg.num_batches} batches of {cfg.batch_size} samples per batch)"
    if cfg.min_cw_errors > 0:
        budget = f"At most {budget}, stopping early at {cfg.min_cw_errors} CW errors"
    log.info(budget)
    log.info(f"Dataloading will use {cfg.num_workers} cpus")
    compiled = getattr(model.decoder, "_compile", False)
    log.info(f"Precision: {cfg.precision}, torch.compile: {compiled}")

    # Resolve HDD correction capability (t=0 if hdd=false)
    t = resolve_hdd_t(model, code, cfg.hdd)
    if cfg.hdd:
        log.info(f"HDD emulation enabled (correction capability t = {t})")

    # Instantiate the test-time scaling strategy (defaults to no-TTS if not set)
    tts = instantiate(cfg.tts)
    tts.validate(model, code)
    if tts.name == "no-tts":
        log.info("No TTS - Standard decoding (single forward pass)")
    else:
        tts_param_str = ""
        if tts.name == "self-boosting":
            tts_param_str = f"with {tts.num_iters} iterations"
        if tts.name == "tta":
            tts_param_str = f"with {tts.num_perms} permutations"
        log.info(
            f"TTS strategy: {tts.name} {tts_param_str} (suffix={tts.suffix or '<none>'})"
        )

    # Build the output file path
    pathlib.Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    # suffix order: <tts>[-bf16][-hdd]
    suffix = (
        tts.suffix
        + ("-bf16" if cfg.precision == "bf16-mixed" else "")
        + ("-hdd" if cfg.hdd else "")
    )
    output_file = cfg.output_dir + "/" + pathlib.Path(model_file).stem + suffix + ".csv"
    log.info(f"Results will be saved to file: {output_file}")

    # Evaluate the model - The results are returned in a list of dicts,
    # using one dict of metrics per Eb/N0 point
    perfs = test_model(
        code,
        model,
        ebno_dB_range,
        output_file,
        tts=tts,
        num_workers=cfg.num_workers,
        test_bs=cfg.batch_size,
        n_test_batches=cfg.num_batches,
        t=t,
        min_cw_errors=cfg.min_cw_errors,
        precision=cfg.precision,
    )

    # Pretty print results in the terminal
    # for some reason, integer numbers are evaluated as float when there are other
    # float columns present, see: https://github.com/astanin/python-tabulate/issues/18
    table = tabulate(
        perfs,  # type: ignore[arg-type]
        headers="keys",
        floatfmt=[".2f", ".4E", ".4E", ".0f", ".0f", ".0f"],
        showindex=False,
    )
    log.info(f"Results:\n{table}\n")


def main() -> None:
    """Console-script entry point.

    For convenience, we want to be able to pass the model checkpoint <ckpt>
    as the first positional argument, without needing to specify `model=<ckpt>`
    explicitly. However Hydra has no notion of positional args, so we have
    to hack around it by manually inspecting sys.argv: we rewrite the first
    positional argument into a `model=<ckpt>` override (see conf/test.yaml)
    before handing off to Hydra. The first arg is treated as a model path when
    it doesn't look like a Hydra override (no '='), an addition/removal
    directive ('+'/'~'), or a flag ('-'/'--'). Everything else is left
    untouched to make sure overriding other parameters still work as expected.
    """
    if len(sys.argv) >= 2:
        first = sys.argv[1]
        # If the first arg doesn't look like a Hydra override, rewrite it
        # as a model path override in the form `model=<ckpt>`.
        if not (first.startswith(("+", "~", "-")) or "=" in first):
            sys.argv[1] = f"model={first}"
    _main()


if __name__ == "__main__":
    main()
