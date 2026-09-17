#!/usr/bin/env python
"""Overlay FER-vs-Eb/N0 curves from sbnd-test CSVs.

python exp/plot_fer.py --out exp/fig.png --title "..." A0=log/test/a.csv A1=log/test/b.csv
"""

import argparse, csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: str) -> tuple[list[float], list[float], list[float]]:
    snr, fer, err = [], [], []
    for r in csv.DictReader(open(path)):
        snr.append(float(r["Eb/N0"]))
        fer.append(float(r["WER"]))
        err.append(float(r["CW errors"]))
    return snr, fer, err


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("curves", nargs="+", help="LABEL=path/to/eval.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--ref", default=None, help="LABEL=path drawn dashed/grey")
    args = p.parse_args()

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True, height_ratios=[3, 1]
    )
    entries = [c.split("=", 1) for c in args.curves]
    base = None  # first curve: reference for the ratio panel
    if args.ref:
        entries.append(args.ref.split("=", 1))
    for i, (label, path) in enumerate(entries):
        snr, fer, err = load(path)
        is_ref = args.ref is not None and i == len(entries) - 1
        # 95% CI on a binomial FER, from the frame-error count of each point
        lo = [f * (1 - 1.96 / e**0.5) for f, e in zip(fer, err)]
        hi = [f * (1 + 1.96 / e**0.5) for f, e in zip(fer, err)]
        style = dict(marker="o", ms=4)
        if is_ref:
            style = dict(color="0.5", ls="--", marker="", lw=1.5, zorder=1)
        (line,) = ax.semilogy(snr, fer, label=label, **style)
        ax.fill_between(snr, lo, hi, color=line.get_color(), alpha=0.2, lw=0)
        # Lower panel: FER relative to the first curve. The differences between
        # variants are a few percent, invisible on a log axis spanning decades.
        if base is None:
            base = dict(zip(snr, fer))
        elif not is_ref:
            r = [f / base[x] for x, f in zip(snr, fer)]
            # 95% CI on the ratio of two independent binomial counts
            e0 = dict(zip(snr, err))
            rerr = [1.96 * (1 / e + 1 / base_err[x]) ** 0.5 for x, e in zip(snr, err)]
            rax.errorbar(
                snr,
                r,
                yerr=[ri * ei for ri, ei in zip(r, rerr)],
                marker="o",
                ms=4,
                capsize=3,
                color=line.get_color(),
                label=label,
            )
        if base is not None and len(entries) and label == entries[0][0]:
            base_err = dict(zip(snr, err))

    ax.set_ylabel("FER")
    ax.set_title(args.title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    rax.axhline(1.0, color="k", lw=0.8)
    rax.set_xlabel("$E_b/N_0$ (dB)")
    rax.set_ylabel(f"FER / {entries[0][0]}")
    rax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
