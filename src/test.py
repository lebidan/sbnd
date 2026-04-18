# Evaluate the test performance of a trained SBND model through Monte Carlo simulations.

import torch, pathlib, argparse, csv

from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore[import-untyped]
from tabulate import tabulate  # type: ignore[import-untyped]

from .utils import get_rank_zero_logger, setup_logging
from .codes import LinearCode
from .model import SBNDLitModule
from .data import OnDemandDataset

log = get_rank_zero_logger(__name__)


def load_lit_model(model_file: str) -> SBNDLitModule:
    return SBNDLitModule.load_from_checkpoint(model_file, weights_only=False)


def count_parameters(model: SBNDLitModule) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def bipolar_to_bit(x: Tensor) -> Tensor:
    return (x < 0).to(torch.int8)


# column labels for the output csv file
COLUMNS = ["Eb/N0", "WER", "BER", "CW errors", "Bit errors", "Total CW"]


def load_csv(path: str) -> list[dict[str, float]]:
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def write_csv(rows: list[dict[str, float]], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def update_error_stats(
    preds: Tensor,
    targets: Tensor,
    syndromes: Tensor,
    i_set: Tensor,
    stats: dict[str, float | int],
    t: int = 0,
) -> None:
    total_cw = targets.size(0)
    if preds.size(1) < targets.size(1):
        assert preds.size(1) == len(i_set)
        targets = targets[
            :, i_set
        ]  # iSBND only estimate the error pattern on the message bits
    stats["Total CW"] += total_cw
    # identify all non-zero target error patterns (the all-zero ones don't even enter the decoder)
    nz_target_idx = torch.any(targets, dim=1).nonzero().squeeze(dim=1)
    # among them, those with zero syndromes (+1 syndromes in bipolar form) are necessarily decoding errors
    zero_synd_idx = (
        torch.all(syndromes[nz_target_idx] > 0, dim=1).nonzero().squeeze(dim=1)
    )
    bit_err = (
        preds[nz_target_idx[zero_synd_idx]] != targets[nz_target_idx[zero_synd_idx]]
    )
    stats["Bit errors"] += bit_err[:, i_set].sum().item()
    stats["CW errors"] += torch.any(bit_err, dim=1).sum().item()
    # analyze the predictions for the non-zero error patterns with a non-zero syndrome
    nz_synd_idx = (
        torch.any(syndromes[nz_target_idx] < 0, dim=1).nonzero().squeeze(dim=1)
    )
    bit_err = preds[nz_target_idx[nz_synd_idx]] != targets[nz_target_idx[nz_synd_idx]]
    # emulate HDD by counting the number of bit errors and declaring a decoding success if 
    # the number of bit errors is less than the error correction capability t of the code
    nb_err = bit_err.sum(dim=1)
    bit_err = bit_err & (nb_err > t).unsqueeze(1)
    stats["Bit errors"] += bit_err[:, i_set].sum().item()
    stats["CW errors"] += torch.any(bit_err, dim=1).sum().item()


def test_model(
    code: LinearCode,
    model: SBNDLitModule,
    ebno_dB_range: Tensor,
    output_file: str,
    test_bs: int = 4096,
    n_test_batches: int = 512,
    num_workers: int = 16,
    show_progress: bool = True,
    t: int = 0,
) -> list[dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    model = model.to(device)
    model.eval()
    # Load existing rows if the output file already exists, otherwise start fresh
    rows: list[dict[str, float]] = []
    if pathlib.Path(output_file).exists():
        rows = load_csv(output_file)
        log.info(
            f"Appending to existing file: {output_file} ({len(rows)} rows already present)"
        )
    # Setup and run MC simulation
    i_set = torch.arange(0, code.k)
    for ebno_dB in ebno_dB_range:
        print(f"Simulating Eb/N0 = {ebno_dB} dB")
        error_stats: dict[str, float] = {
            "Eb/N0": ebno_dB.item(),
            "WER": 0.0,
            "BER": 0.0,
            "CW errors": 0.0,
            "Bit errors": 0.0,
            "Total CW": 0.0,
        }
        ds = OnDemandDataset(
            code, ebno_dB=ebno_dB, n_batches=n_test_batches, bs=test_bs, train=False
        )
        dl = DataLoader(ds, batch_size=None, num_workers=num_workers)
        with torch.no_grad():
            for batch in tqdm(dl, disable=not show_progress):
                ym, syndromes, targets = batch
                logits = model(ym.to(device), syndromes.to(device))
                preds = bipolar_to_bit(logits)
                update_error_stats(
                    preds.cpu(), targets, syndromes, i_set, error_stats, t
                )
        # collect error stats in a list of dicts
        error_stats["WER"] = error_stats["CW errors"] * 1.0 / error_stats["Total CW"]
        error_stats["BER"] = error_stats["Bit errors"] / (
            error_stats["Total CW"] * len(i_set)
        )
        # append new SNR point, then deduplicate by Eb/N0 keeping the latest value
        rows.append(error_stats)
        deduped: dict[float, dict[str, float]] = {}
        for row in rows:
            deduped[row["Eb/N0"]] = row
        rows = list(deduped.values())
        # print error stats and save to csv after each SNR point
        print(error_stats)
        write_csv(rows, output_file)
    return rows


def main() -> None:

    setup_logging()  # mimic hydra's nice colorful logging style in the terminal

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        prog="test", description="Evaluate the FER & BER performance of a trained model"
    )
    parser.add_argument("model", type=str, help="model checkpoint")
    parser.add_argument(
        "--code",
        type=str,
        help="code .mat file path (overrides the path stored in the checkpoint)",
        default=None,
    )
    parser.add_argument(
        "--output", type=str, help="path to output csv file", default="./log/test"
    )
    parser.add_argument(
        "--snr_min", type=float, help="minimum Eb/N0 value to simulate", default=0.0
    )
    parser.add_argument(
        "--snr_max", type=float, help="maximum Eb/N0 value to simulate", default=5.0
    )
    parser.add_argument(
        "--snr_step", type=float, help="step between Eb/N0 values", default=1.0
    )
    parser.add_argument("--batch_size", type=int, help="test batch size", default=4096)
    parser.add_argument(
        "--num_batches",
        type=int,
        help="number of batches per Eb/N0 value",
        default=1024,
    )
    parser.add_argument(
        "--num_workers", type=int, help="number of workers for dataloading", default=8
    )
    parser.add_argument(
        "--hdd",
        action="store_true",
        help="emulate hard-decision decoding (perfect correction if errors <= t)",
        default=False,
    )
    args = parser.parse_args()

    # Load model first (code path is stored in its hparams)
    model_file = args.model
    if not model_file.endswith(".ckpt"):
        model_file += ".ckpt"
    log.info(f"Loading model from file: {model_file}")
    model = load_lit_model(model_file)
    log.info(f"Model {model} has been successfully loaded")
    log.info(f"This model has {count_parameters(model):,} trainable parameters")

    # Resolve code file: explicit --code flag > path stored in checkpoint
    if args.code is not None:
        code_file = args.code
    else:
        code_file = model.hparams.code_path  # type: ignore[attr-defined]
        if not code_file:
            raise ValueError(
                "No code path found in checkpoint hparams. "
                "Use --code to specify the code file explicitly."
            )
    if not code_file.endswith(".mat"):
        code_file += ".mat"
    log.info(f"Loading code from file: {code_file}")
    code = LinearCode(code_file)
    log.info(f"Code {code} has been successfully loaded")

    # Load simulation parameters
    ebno_dB_range = torch.arange(
        args.snr_min, args.snr_max + args.snr_step, args.snr_step
    )
    log.info(
        f"Eb/N0 range to simulate: from {ebno_dB_range[0]} to {ebno_dB_range[-1]} by step of {args.snr_step} dB ({len(ebno_dB_range)} values)"
    )
    log.info(
        f"{args.num_batches * args.batch_size:,} samples per Eb/N0 value ({args.num_batches} batches of {args.batch_size} samples per batch)"
    )
    log.info(f"Dataloading will use {args.num_workers} cpus")
    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    suffix = "-hdd" if args.hdd else ""
    output_file = args.output + "/" + pathlib.Path(model_file).stem + suffix + ".csv"
    log.info(f"Results will be saved to file: {output_file}")

    t = (code.dmin - 1) // 2 if args.hdd else 0
    if args.hdd:
        log.info(f"HDD emulation enabled (correction capability t = {t})")

    # Evaluate the model - The results are returned in a list of dicts,
    # using one dict of metrics per Eb/N0 point
    perfs = test_model(
        code,
        model,
        ebno_dB_range,
        output_file,
        num_workers=args.num_workers,
        test_bs=args.batch_size,
        n_test_batches=args.num_batches,
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
