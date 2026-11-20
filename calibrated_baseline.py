"""E0.5/E1: data-calibrated MWPM, the honest classical adversary.

Fits matching-graph edge weights from sampled syndrome data alone
(Spitz et al. arXiv:1712.02360): for each interior edge (i,j) the defect
co-occurrence rate identifies its error probability via

    p_e ≈ (P_ij − m_i·m_j) / (1 − m_i − m_j + P_ij)

which is exact in the single-edge limit and a good approximation when
marginals stay small. Boundary edges (single-detector) can't be estimated
from co-occurrence and keep nominal weights (noted limitation).

Also injects biased noise: stim has no bias parameter on generated
circuits, so DEPOLARIZE1(p) is text-replaced with a biased
PAULI_CHANNEL_1(px, py, pz) preserving total rate:
    px = p·η/(η+2),  py = pz = p/(η+2)      (η = X:other ratio)

Three decoders on identical shots:
  naive      : MWPM on the nominal depolarising DEM
  calibrated : MWPM re-weighted from the sampled data
  informed   : MWPM on the true generating DEM (oracle upper bound)
"""

import argparse
import re

import networkx as nx
import numpy as np
import pymatching
import stim

SEED = 42


def build_circuit(distance, rounds, noise, memory="z"):
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{memory}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise,
        before_round_data_depolarization=noise,
        before_measure_flip_probability=noise,
        after_reset_flip_probability=noise,
    )


def biased_circuit(distance, rounds, noise, eta, memory="z", hetero=False):
    """Replace DEPOLARIZE1 with biased PAULI_CHANNEL_1 at total rate p.
    hetero=False: uniform X-bias (all edges shift identically, expected
    to NOT hurt MWPM, verified). hetero=True: alternate X-bias/Z-bias per
    qubit, a structural mismatch where uniform weights are relatively
    wrong edge-to-edge."""
    px = noise * eta / (eta + 2)
    po = noise / (eta + 2)
    text = str(build_circuit(distance, rounds, noise, memory))
    n_sub = 0

    def repl(m):
        nonlocal n_sub
        n_sub += 1
        p_str, targets = m.group(1), m.group(2).split()
        if not hetero:
            return f"PAULI_CHANNEL_1({px},{po},{po}) " + " ".join(targets)
        lines = []
        for i, q in enumerate(targets):
            if i % 2 == 0:   # X-biased qubits
                lines.append(f"PAULI_CHANNEL_1({px},{po},{po}) {q}")
            else:            # Z-biased qubits
                lines.append(f"PAULI_CHANNEL_1({po},{po},{px}) {q}")
        return "\n        ".join(lines)

    text = re.sub(r"DEPOLARIZE1\(([^)]+)\)([ \d]+)", repl, text)
    print(f"bias injection: {n_sub} DEPOLARIZE1 -> PAULI_CHANNEL_1 "
          f"(px={px:.4f}, py=pz={po:.5f}, eta={eta}, hetero={hetero})")
    return stim.Circuit(text)


def sample(circuit, shots):
    sampler = circuit.compile_detector_sampler()
    det, obs = sampler.sample(shots, separate_observables=True)
    return det, obs[:, 0]


def mwpm(circuit, det, obs):
    dem = circuit.detector_error_model(decompose_errors=True)
    m = pymatching.Matching.from_detector_error_model(dem)
    pred = m.decode_batch(det.astype(bool))[:, 0]
    return float((pred != obs.astype(bool)).mean())


def calibrated_matching(nominal_circuit, det):
    """Rebuild Matching with edge weights fitted from syndrome data."""
    dem = nominal_circuit.detector_error_model(decompose_errors=True)
    m_nom = pymatching.Matching.from_detector_error_model(dem)
    G = nx.Graph(m_nom.to_networkx())

    n_det = det.shape[1]
    marg = det.mean(axis=0)
    n_fit = n_skip = 0
    for u, v, d in G.edges(data=True):
        if u >= n_det or v >= n_det:          # boundary virtual node
            n_skip += 1
            continue
        mi, mj = marg[u], marg[v]
        pij = float(np.logical_and(det[:, u], det[:, v]).mean())
        denom = 1.0 - mi - mj + pij
        pe = (pij - mi * mj) / denom if denom > 1e-9 else mi * mj
        pe = np.clip(pe, 1e-6, 0.49)
        d["weight"] = float(np.log((1.0 - pe) / pe))
        n_fit += 1
    print(f"calibration: fitted {n_fit} interior edges "
          f"({n_skip} boundary edges kept nominal)")
    return pymatching.Matching(G)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--distance", type=int, default=3)
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--noise", type=float, default=0.005)
    p.add_argument("--eta", type=float, default=100.0,
                   help="bias ratio (1.0 = depolarising sanity check)")
    p.add_argument("--hetero", action="store_true",
                   help="alternate X/Z bias per qubit = structural mismatch")
    p.add_argument("--memory", default="z", choices=["x", "z"])
    p.add_argument("--shots", type=int, default=200_000)
    args = p.parse_args()

    np.random.seed(SEED)
    nominal = build_circuit(args.distance, args.rounds, args.noise,
                            memory=args.memory)
    truth = (nominal if args.eta == 1.0 else
             biased_circuit(args.distance, args.rounds, args.noise,
                            args.eta, memory=args.memory,
                            hetero=args.hetero))

    det, obs = sample(truth, args.shots)
    split = args.shots // 2
    det_cal, det_te = det[:split], det[split:]
    obs_te = obs[split:]

    print(f"\nd={args.distance} r={args.rounds} p={args.noise} eta={args.eta} "
          f"| {split} calibration + {split} test shots\n")

    naive = mwpm(nominal, det_te, obs_te)
    m_cal = calibrated_matching(nominal, det_cal)
    pred_cal = m_cal.decode_batch(det_te.astype(bool))[:, 0]
    cal_err = float((pred_cal != obs_te.astype(bool)).mean())
    informed = mwpm(truth, det_te, obs_te)

    print(f"naive MWPM (depolarising prior):   {naive:.4f}")
    print(f"calibrated MWPM (fitted from data):{cal_err:>9.4f}")
    print(f"informed MWPM (oracle true DEM):   {informed:.4f}")
    print(f"\ncalibration recovers {100 * (naive - cal_err) / max(naive - informed, 1e-9):.0f}% "
          f"of the naive->oracle gap")


if __name__ == "__main__":
    main()
