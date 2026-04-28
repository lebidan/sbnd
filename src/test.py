# Evaluate the test performance of a trained SBND model through Monte Carlo simulations.

import os, csv, pathlib
import torch, hydra

from hydra.utils import instantiate
from torch import Tensor
from torch.utils.data import DataLoader
from omegaconf import DictConfig
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


def update_error_stats(
    code: LinearCode,
    error_space: str,
    preds: Tensor,
    targets: Tensor,
    syndromes: Tensor,
    stats: dict[str, float | int],
    t: int = 0,
) -> None:
    """
    Accumulate codeword (frame) and bit error counts over a test batch.

    BER is always reported on the k message bits. CW errors are reported on the
    full n-bit codeword when `error_space == "codeword"` (true FER), and on the
    k message bits otherwise (we don't have access to codeword-level errors when
    working in message mode).
    """
    total_cw = targets.size(0)
    stats["Total CW"] += total_cw
    # bit-level diff in the space where the model was trained (shape (bs, n) or (bs, k))
    # and in the message space (always (bs, k)) for BER counting
    diff_fer = (preds != targets).to(torch.int8)
    if error_space == "codeword":
        diff_ber = (diff_fer @ code.Ginv).bitwise_and(1)
    else:
        diff_ber = diff_fer
    # identify all non-zero target error patterns (the all-zero ones don't even enter the decoder)
    nz_target_idx = torch.any(targets, dim=1).nonzero().squeeze(dim=1)
    # among them, those with zero syndromes (+1 syndromes in bipolar form) are necessarily decoding errors
    zero_synd_idx = (
        torch.all(syndromes[nz_target_idx] > 0, dim=1).nonzero().squeeze(dim=1)
    )
    zs = nz_target_idx[zero_synd_idx]
    stats["Bit errors"] += diff_ber[zs].sum().item()
    stats["CW errors"] += torch.any(diff_fer[zs], dim=1).sum().item()
    # analyze the predictions for the non-zero error patterns with a non-zero syndrome
    nz_synd_idx = (
        torch.any(syndromes[nz_target_idx] < 0, dim=1).nonzero().squeeze(dim=1)
    )
    ns = nz_target_idx[nz_synd_idx]
    fer_err = diff_fer[ns]
    ber_err = diff_ber[ns]
    # emulate HDD by counting the number of bit errors and declaring a decoding success if
    # the number of bit errors is less than the error correction capability t of the code
    if t > 0:
        hdd_miss = (fer_err.sum(dim=1) > t).unsqueeze(1)
        fer_err = fer_err & hdd_miss
        ber_err = ber_err & hdd_miss
    stats["Bit errors"] += ber_err.sum().item()
    stats["CW errors"] += torch.any(fer_err, dim=1).sum().item()


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
) -> list[dict[str, float]]:
    if tts is None:
        tts = SingleShotDecoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    model = model.to(device)
    model.eval()
    error_space = getattr(model.decoder, "error_space", "codeword")
    # Load existing rows if the output file already exists, otherwise start fresh.
    # Index by Eb/N0 so we can accumulate stats when re-running the same SNR point.
    # Keys are rounded to SNR_KEY_DECIMALS to absorb fp drift between torch.arange
    # accumulation and CSV float round-trips (matters for steps like 0.1 or 0.05).
    stats_by_snr: dict[float, dict[str, float]] = {}
    if pathlib.Path(output_file).exists():
        for row in load_csv(output_file):
            row["Eb/N0"] = _snr_key(row["Eb/N0"])
            stats_by_snr[row["Eb/N0"]] = row
        log.info(
            f"Appending to existing file: {output_file} ({len(stats_by_snr)} rows already present)"
        )
    # Setup and run MC simulation
    for ebno_dB in ebno_dB_range:
        snr = _snr_key(ebno_dB.item())
        print(f"Simulating Eb/N0 = {ebno_dB} dB")
        # Seed counters from any existing row at this SNR so new samples
        # accumulate on top instead of replacing it.
        prev = stats_by_snr.get(snr)
        error_stats: dict[str, float] = {
            "Eb/N0": snr,
            "WER": 0.0,
            "BER": 0.0,
            "CW errors": prev["CW errors"] if prev is not None else 0.0,
            "Bit errors": prev["Bit errors"] if prev is not None else 0.0,
            "Total CW": prev["Total CW"] if prev is not None else 0.0,
        }
        if prev is not None:
            log.info(
                f"Cumulating with existing stats at Eb/N0={snr} dB "
                f"({int(prev['Total CW']):,} CW already simulated)"
            )
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
            for batch in tqdm(dl, disable=not show_progress):
                ym, syndromes, targets = batch
                ym_dev = ym.to(device)
                synd_dev = syndromes.to(device)
                targets_dev = targets.to(device)
                preds = tts.decode(model, code, ym_dev, synd_dev, targets_dev)  # type: ignore[attr-defined]
                update_error_stats(
                    code, error_space, preds.cpu(), targets, syndromes, error_stats, t
                )
        # recompute WER/BER from the cumulative totals
        error_stats["WER"] = error_stats["CW errors"] * 1.0 / error_stats["Total CW"]
        error_stats["BER"] = error_stats["Bit errors"] / (
            error_stats["Total CW"] * code.k
        )
        stats_by_snr[snr] = error_stats
        # print error stats and save to csv after each SNR point
        print(error_stats)
        write_csv(list(stats_by_snr.values()), output_file)
    return list(stats_by_snr.values())


# conf/ is not part of the installed package; it lives in the project root.
# sbnd-test must be run from the directory that contains conf/.
_conf_dir = os.path.join(os.getcwd(), "conf")


@hydra.main(version_base="1.3", config_path=_conf_dir, config_name="test")
def main(cfg: DictConfig) -> None:

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
    log.info(
        f"{cfg.num_batches * cfg.batch_size:,} samples per Eb/N0 value ({cfg.num_batches} batches of {cfg.batch_size} samples per batch)"
    )
    log.info(f"Dataloading will use {cfg.num_workers} cpus")

    # Resolve HDD correction capability (t=0 if hdd=false)
    t = resolve_hdd_t(model, code, cfg.hdd)
    if cfg.hdd:
        log.info(f"HDD emulation enabled (correction capability t = {t})")

    # Instantiate the test-time scaling strategy (defaults to single-shot)
    tts = instantiate(cfg.tts)
    tts.validate(model, code)
    log.info(f"TTS strategy: {tts.name} (suffix={tts.suffix or '<none>'})")

    # Build the output file path
    pathlib.Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    suffix = tts.suffix + ("-hdd" if cfg.hdd else "")
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


if __name__ == "__main__":
    main()
