"""Watch the decoder work — surface-code syndrome visualization.

Panel grid: one subplot per syndrome round (detector time coordinate),
showing the code lattice, which detectors fired in one sampled shot, and
the error-pairs MWPM matched (lines) or boundary matches (open markers).
Bottom: live naive vs calibrated vs informed LER bar chart.

Run: .venv/bin/python viz_decoder.py [--noise 0.02 --eta 100 --hetero]
"""

import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pymatching
import stim

from calibrated_baseline import (biased_circuit, build_circuit,
                                 calibrated_matching, sample)

SEED = 42


def detector_coords(circuit):
    coords = circuit.get_detector_coordinates()
    return {i: c for i, c in coords.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--distance", type=int, default=3)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--hetero", action="store_true")
    p.add_argument("--eval-shots", type=int, default=20_000)
    args = p.parse_args()
    np.random.seed(SEED)

    nominal = build_circuit(args.distance, args.rounds, args.noise)
    truth = (nominal if args.eta == 1.0 else
             biased_circuit(args.distance, args.rounds, args.noise,
                            args.eta, hetero=args.hetero))
    coords = detector_coords(truth)
    times = sorted({c[2] for c in coords.values()})

    dem = truth.detector_error_model(decompose_errors=True)
    m = pymatching.Matching.from_detector_error_model(dem)

    # pick a shot with observable flip for an interesting decode
    det, obs = sample(truth, args.eval_shots + 200)
    flips = np.flatnonzero(obs[:200])
    idx = int(flips[0]) if len(flips) else int(np.argmax(det[:200].sum(1)))
    det_one, obs_one = det[idx], obs[idx]
    pairs = m.decode_to_matched_dets_array(det_one.astype(bool))

    # single projected lattice: defects labeled by round, matched pairs
    # drawn on (x,y) regardless of t (space-time edges are the norm)
    fig = plt.figure(figsize=(14, 6))
    ax_lat = fig.add_axes([0.03, 0.08, 0.55, 0.78])
    ax_bar = fig.add_axes([0.68, 0.15, 0.28, 0.68])

    row = det_one.astype(bool)
    fired = [i for i in coords if row[i]]
    lat = {(c[0], c[1]) for c in coords.values()}
    ax_lat.scatter([p[0] for p in lat], [p[1] for p in lat],
                   s=45, c="#d4d9e0", zorder=1)
    for a, b in pairs:
        if a < 0 or b < 0:                    # boundary match (-1)
            i = a if a >= 0 else b
            if i in coords:
                ax_lat.scatter([coords[i][0]], [coords[i][1]], s=170,
                               facecolors="none", edgecolors="#dc2626",
                               linewidths=2.2, zorder=4)
        elif a in coords and b in coords:
            ax_lat.plot([coords[a][0], coords[b][0]],
                        [coords[a][1], coords[b][1]],
                        color="#dc2626", lw=2.2, alpha=0.85, zorder=2)
    ax_lat.scatter([coords[i][0] for i in fired],
                   [coords[i][1] for i in fired],
                   s=110, facecolors="#7c3aed", edgecolors="k", zorder=3)
    for i in fired:
        ax_lat.annotate(f"t{int(coords[i][2])}",
                        (coords[i][0], coords[i][1]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=9, color="#5a6270")
    ax_lat.set_aspect("equal")
    ax_lat.set_title("syndrome lattice — defects (purple, labeled by round),\n"
                     "MWPM matched pairs (red), boundary matches (rings)",
                     fontsize=11)
    ax_lat.axis("off")

    n_te = args.eval_shots // 2
    det_cal, det_te = det[:n_te], det[n_te:args.eval_shots]
    obs_te = obs[n_te:args.eval_shots]
    naive = float((m.decode_batch(det_te.astype(bool))[:, 0]
                   != obs_te.astype(bool)).mean())
    m_cal = calibrated_matching(nominal, det_cal)
    cal = float((m_cal.decode_batch(det_te.astype(bool))[:, 0]
                 != obs_te.astype(bool)).mean())
    bars = ax_bar.bar(["naive\nMWPM", "calibrated\nMWPM"],
                      [naive, cal], color=["#5a6270", "#7c3aed"])
    ax_bar.bar_label(bars, fmt="%.3f")
    ax_bar.set_ylabel("logical error rate")
    ax_bar.set_title(f"LER on {n_te:,} test shots", fontsize=10)
    ax_bar.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"rotated memory-Z d={args.distance} r={args.rounds} "
        f"p={args.noise} eta={args.eta} hetero={args.hetero} — one shot decode",
        fontsize=12)
    fig.tight_layout()
    out = "results/decoder_viz.png"
    import os
    os.makedirs("results", exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
