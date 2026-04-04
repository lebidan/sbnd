import torch, pathlib, argparse
import pandas as pd  # type: ignore[import-untyped]

from torch import Tensor
from pandas import DataFrame  # type: ignore[import-untyped]
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


def update_error_stats(
    preds: Tensor,
    targets: Tensor,
    syndromes: Tensor,
    i_set: Tensor,
    stats: dict[str, float | int],
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
) -> DataFrame:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    # Setup output dataframe & file
    df = pd.DataFrame(
        columns=["Eb/N0", "WER", "BER", "CW errors", "Bit errors", "Total CW"]
    )
    # Setup and run MC simulation
    i_set = torch.arange(0, code.k)
    for ebno_dB in ebno_dB_range:
        print(f"Simulating Eb/N0 = {ebno_dB} dB")
        error_stats = {
            "Eb/N0": ebno_dB.item(),
            "WER": 0.0,
            "BER": 0.0,
            "CW errors": 0,
            "Bit errors": 0,
            "Total CW": 0,
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
                update_error_stats(preds.cpu(), targets, syndromes, i_set, error_stats)
        # save error stats in dataframe & write them also to csv file
        error_stats["WER"] = error_stats["CW errors"] * 1.0 / error_stats["Total CW"]
        error_stats["BER"] = (
            error_stats["Bit errors"] * 1.0 / (error_stats["Total CW"] * len(i_set))
        )
        df.loc[len(df)] = error_stats
        print(error_stats)
        df.to_csv(output_file, index=False)
    return df


def main() -> None:

    setup_logging()

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
        "--num_batch", type=int, help="number of batches per Eb/N0 value", default=1024
    )
    parser.add_argument(
        "--num_workers", type=int, help="number of workers for dataloading", default=8
    )

    args = parser.parse_args()

    # Load model first (code path is stored in its hparams)

    model_file = args.model
    if not model_file.endswith(".ckpt"):
        model_file += ".ckpt"
    log.info(f"Loading model from file: {model_file}")
    model = load_lit_model(model_file)
    log.info(f"Model {model} has been successfully loaded")
    log.info(f"This model has {count_parameters(model):,} trainable parameters\n")

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
    log.info(f"Code {code} has been successfully loaded\n")

    # Load simulation parameters

    ebno_dB_range = torch.arange(
        args.snr_min, args.snr_max + args.snr_step, args.snr_step
    )
    log.info(
        f"Eb/N0 range to simulate: from {ebno_dB_range[0]} to {ebno_dB_range[-1]} by step of {args.snr_step} dB ({len(ebno_dB_range)} values)"
    )
    log.info(
        f"{args.num_batch * args.batch_size:,} samples per Eb/N0 value ({args.num_batch} batches of {args.batch_size} samples per batch)"
    )
    log.info(f"Dataloading will use {args.num_workers} cpus")

    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    output_file = args.output + "/" + pathlib.Path(model_file).stem + ".csv"
    log.info(f"Results will be saved to file: {output_file}\n")

    # Evaluate the model and save results in a dataframe

    df = test_model(
        code,
        model,
        ebno_dB_range,
        output_file,
        num_workers=args.num_workers,
        test_bs=args.batch_size,
        n_test_batches=args.num_batch,
    )
    log.info("")

    # Pretty print results in the terminal
    # for some weird reason, integer numbers are still evaluated as float when there are other float columns
    # in pandas DataFrames, see: https://github.com/astanin/python-tabulate/issues/18

    table = tabulate(
        df,  # type: ignore[arg-type]
        headers="keys",
        floatfmt=[".2f", ".4E", ".4E", ".0f", ".0f", ".0f"],
        showindex=False,
    )
    log.info(f"Results:\n{table}\n")


if __name__ == "__main__":
    main()
