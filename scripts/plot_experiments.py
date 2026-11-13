"""FER and BER vs Eb/N0 for every code in docs/experiments.md.

Reads the eval-log tables of docs/experiments.md, keeps the best eval per model
(decoder + size), and writes docs/figs/<code>.png next to the hard-coded MLD curve.
Run from the repo root: python scripts/plot_experiments.py
"""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "experiments.md"
OUT = ROOT / "docs" / "figs"

# MLD reference curves: (label, Eb/N0 dB, FER, info BER).
# From sdd-julia OSD runs (>= 100 frame errors per point) unless noted.
# fmt: off
MLD = {
    "BCH (31, 21, 5)": (
        "MLD (OSD order 21, exact)",
        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5],
        [0.4798, 0.3752, 0.2768, 0.1851, 0.1146, 0.06315, 0.03135, 0.01347, 0.0056, 0.00177, 0.0004787, 0.0001181, 2.381e-05, 4.401e-06, 5.623e-07, 6.016e-08],
        [0.101, 0.0776, 0.05615, 0.03696, 0.02249, 0.01223, 0.00596, 0.002538, 0.001033, 0.0003243, 8.305e-05, 2.018e-05, 3.991e-06, 7.315e-07, 9.8e-08, 9.913e-09],
    ),
    "BCH (63, 24, 15)": (
        "MLD (OSD order 4)",
        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6],
        [0.2629, 0.1657, 0.09372, 0.0457, 0.01923, 0.006585, 0.00197, 0.0005, 0.0001056, 1.211e-05, 1.732e-06, 1.295e-07, 1.01e-08],
        [0.07908, 0.04907, 0.02737, 0.01319, 0.005433, 0.001853, 0.000541, 0.0001354, 2.717e-05, 3.045e-06, 4.672e-07, 3.283e-08, 2.7e-09],
    ),
    "BCH (63, 30, 13)": (
        "MLD (OSD order 4)",
        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5],
        [0.3372, 0.2172, 0.1225, 0.05878, 0.02355, 0.007812, 0.002052, 0.0004115, 6.13e-05, 8.633e-06, 7.936e-07, 6.507e-08],
        [0.09096, 0.05768, 0.03214, 0.0152, 0.00603, 0.00196, 0.0005108, 0.0001003, 1.504e-05, 2.106e-06, 1.853e-07, 1.433e-08],
    ),
    "BCH (63, 36, 11)": (
        "MLD (OSD order 3)",
        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6],
        [0.4621, 0.3186, 0.1928, 0.1002, 0.04327, 0.01524, 0.004188, 0.0008125, 0.0001759, 1.796e-05, 1.963e-06, 1.201e-07, 6.59e-09],
        [0.1051, 0.07115, 0.04231, 0.02164, 0.00924, 0.003202, 0.0008625, 0.0001642, 3.501e-05, 3.589e-06, 3.744e-07, 2.229e-08, 1.192e-09],
    ),
    "BCH (63, 39, 9)": (
        "MLD (OSD order 3)",
        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6],
        [0.5449, 0.3952, 0.2545, 0.1414, 0.06645, 0.02554, 0.00788, 0.001917, 0.0003976, 5.58e-05, 6.12e-06, 6.203e-07, 5.032e-08],
        [0.1107, 0.07823, 0.04919, 0.02676, 0.01229, 0.004613, 0.001399, 0.0003304, 6.989e-05, 9.167e-06, 9.717e-07, 9.469e-08, 7.655e-09],
    ),
    "BCH (63, 45, 7)": (
        "MLD (OSD order 2)",
        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5],
        [0.7109, 0.5715, 0.4168, 0.2666, 0.1484, 0.06837, 0.02604, 0.007981, 0.002017, 0.0003559, 6.13e-05, 6.914e-06, 6.758e-07, 4.668e-08],
        [0.1167, 0.09024, 0.06356, 0.03945, 0.02137, 0.00959, 0.003572, 0.001071, 0.000264, 4.394e-05, 7.496e-06, 8.645e-07, 7.629e-08, 5.297e-09],
    ),
    "BCH (63, 51, 5)": (
        "MLD (OSD order 2)",
        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5],
        [0.856, 0.7577, 0.6247, 0.4681, 0.3121, 0.1801, 0.08718, 0.03567, 0.01187, 0.003109, 0.0006944, 0.0001176, 1.953e-05, 2.245e-06, 2.294e-07, 1.649e-08],
        [0.1126, 0.09426, 0.07405, 0.05317, 0.03417, 0.01912, 0.008987, 0.003565, 0.001154, 0.0002936, 6.328e-05, 1.052e-05, 1.675e-06, 1.952e-07, 1.946e-08, 1.362e-09],
    ),
    "eBCH (32, 16, 8)": (
        "MLD (OSD order 16, exact)",
        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5],
        [0.3176, 0.2316, 0.1582, 0.09904, 0.05553, 0.02915, 0.01294, 0.00556, 0.00195, 0.0005642, 0.0001335, 2.481e-05, 5.069e-06, 7.321e-07, 9.044e-08, 5.336e-09],
        [0.08821, 0.06359, 0.04274, 0.02646, 0.01461, 0.007561, 0.003295, 0.001387, 0.0005075, 0.0001439, 3.321e-05, 6.528e-06, 1.251e-06, 1.839e-07, 2.312e-08, 1.354e-09],
    ),
    "Reed-Muller (32, 16, 8)": (
        "MLD (OSD order 16, exact)",
        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7],
        [0.318, 0.2313, 0.1571, 0.09945, 0.05559, 0.02834, 0.01341, 0.00544, 0.00176, 0.0005587, 0.0001284, 2.666e-05, 5.467e-06, 7.881e-07, 8.043e-08],
        [0.1323, 0.09566, 0.06466, 0.04053, 0.02259, 0.0113, 0.005366, 0.002164, 0.0006988, 0.0002294, 5.079e-05, 1.008e-05, 2.17e-06, 2.926e-07, 3.192e-08],
    ),
    "QC-LDPC (96, 48, 10) from RPTU": (
        "MLD (OSD order 4)",
        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5],
        [0.4486, 0.285, 0.1552, 0.06691, 0.02363, 0.00627, 0.00132, 0.0003145, 3.823e-05, 5.634e-06, 6.638e-07, 8.192e-08],
        [0.1001, 0.06165, 0.03238, 0.01355, 0.004602, 0.001146, 0.0002369, 4.809e-05, 5.471e-06, 6.984e-07, 8.256e-08, 9.523e-09],
    ),
    "QC-LDPC (128, 64, 14) from CCSDS": (
        "MLD (OSD order 5)",
        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
        [0.4366, 0.2475, 0.1106, 0.0362, 0.00824, 0.001219, 0.0001315, 9.058e-06, 3.537e-07, 1.899e-08],
        [0.1019, 0.05655, 0.02463, 0.007945, 0.00175, 0.0002537, 2.498e-05, 1.677e-06, 6.063e-08, 2.549e-09],
    ),
    # Older MATLAB record (exp/figs_github.m): SCL decoding with list size 32.
    "Polar (128, 64, 8) from RPTU": (
        "MLD (SCL, L = 32)",
        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6],
        [5.1089e-01, 3.2686e-01, 1.7299e-01, 7.5810e-02, 2.7620e-02, 8.2800e-03, 2.2400e-03, 7.6531e-04, 1.6741e-04, 4.9213e-05, 1.2869e-05, 2.5784e-06, 5.2448e-07],
        [1.9873e-01, 1.1961e-01, 5.8547e-02, 2.3363e-02, 7.3683e-03, 1.8580e-03, 3.7672e-04, 1.0770e-04, 1.8101e-05, 4.9264e-06, 1.2990e-06, 2.5274e-07, 5.4633e-08],
    ),
}
# fmt: on


def wilson(k, n, z=1.96):
    """95 % Wilson interval of a binomial proportion k / n."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def read_csv(path):
    with open(ROOT / path) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def load_curve(log, strat):
    """Monte-Carlo points, extended past their last Eb/N0 by the stratified ones."""
    rows = read_csv(log)
    snr = [r["Eb/N0"] for r in rows]
    fer = [r["WER"] for r in rows]
    ber = [r["BER"] for r in rows]
    lo, hi = zip(*(wilson(r["CW errors"], r["Total CW"]) for r in rows))
    lo, hi = list(lo), list(hi)
    if strat:
        for r in read_csv(strat):
            if r["Eb/N0"] > snr[-1]:
                snr.append(r["Eb/N0"])
                fer.append(r["WER"])
                ber.append(r["BER"])
                lo.append(max(r["WER"] - r["ci95"], r["WER"] * 1e-3))
                hi.append(r["WER"] + r["ci95"])
    c = {
        k: np.array(v) for k, v in dict(snr=snr, fer=fer, ber=ber, lo=lo, hi=hi).items()
    }
    c["n_mc"] = len(rows)
    return c


def parse_doc():
    """{code heading: [row dict, ...]} from the eval-log tables of the doc."""
    codes, code = {}, None
    for line in DOC.read_text().splitlines():
        if line.startswith("## ") and line[3:] != "Contents":
            code = line[3:].strip()
            codes[code] = []
        m = re.match(
            r"\| \[(.+?)\]\(.*?\) \| (.+?) \| (.+?) \| (\d+) \| (.*) \|$", line
        )
        if code and m:
            dec, size, data, epochs, cell = m.groups()
            log = re.search(r"\]\(\.\./(log/test/[^)]+\.csv)\)", cell)
            strat = re.search(r"\[stratified\]\(\.\./(log/test/[^)]+\.csv)\)", cell)
            codes[code].append(
                dict(
                    dec=dec,
                    size=size,
                    data=data,
                    epochs=epochs,
                    log=log.group(1),
                    strat=strat and strat.group(1),
                )
            )
    return {k: v for k, v in codes.items() if v}


def score(curve, snrs):
    """Mean log10 FER over the given Eb/N0 points (lower is better)."""
    m = np.isin(curve["snr"][: curve["n_mc"]], snrs)
    return np.mean(np.log10(np.maximum(curve["fer"][: curve["n_mc"]][m], 1e-12)))


def best_per_model(rows):
    """Best eval per (decoder, size), scored on the Monte-Carlo Eb/N0 points they share."""
    models = {}
    for r in rows:
        r["curve"] = load_curve(r["log"], r["strat"])
        models.setdefault((r["dec"], r["size"]), []).append(r)
    best = []
    for runs in models.values():
        shared = set.intersection(
            *(set(r["curve"]["snr"][: r["curve"]["n_mc"]]) for r in runs)
        )
        best.append(min(runs, key=lambda r: score(r["curve"], list(shared))))
    return best


def label(r):
    data = r["data"].replace(" ML samples", " ML").replace(" at ", " @ ")
    if "ML" in data:
        data = (
            data.replace(", with augmentation", ", aug")
            if "augmentation" in data
            else data + ", no aug"
        )
    return f"{r['dec']} {r['size']} | {data} | {r['epochs']} ep"


def slug(code):
    return re.sub(r"[^a-z0-9]+", "-", code.lower()).strip("-")


def plot(code, runs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.tab10.colors
    xmin = min(r["curve"]["snr"].min() for r in runs)
    xmax = max(r["curve"]["snr"].max() for r in runs)
    for i, r in enumerate(runs):
        c = r["curve"]
        for ax, y in zip(axes, ("fer", "ber")):
            # ponytail: BER band reuses the FER relative interval (bit errors come in frame bursts, so a bit-level binomial CI would be far too narrow)
            s = c[y] / c["fer"]
            ax.semilogy(c["snr"], c[y], "o-", ms=4, color=colors[i], label=label(r))
            ax.fill_between(
                c["snr"], c["lo"] * s, c["hi"] * s, color=colors[i], alpha=0.2, lw=0
            )
    if code in MLD:
        name, snr, fer, ber = MLD[code]
        snr = np.array(snr)
        m = (snr >= xmin) & (snr <= xmax)
        for ax, y in zip(axes, (fer, ber)):
            ax.semilogy(snr[m], np.array(y)[m], "k-", lw=2, label=name)
    for ax, name in zip(axes, ("FER", "BER")):
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel(name)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(code)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{slug(code)}.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    assert np.allclose(wilson(50, 100), (0.4038, 0.5962), atol=1e-4)
    for code, rows in parse_doc().items():
        plot(code, best_per_model(rows))
        print(f"{slug(code)}.png", "" if code in MLD else "(no MLD record)")
