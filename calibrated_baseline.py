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


def burst_circuit(distance, rounds, noise, p_burst, memory="z"):
    """Correlated multi-qubit faults: the regime the matching graph
    cannot represent. Inserts CORRELATED_ERROR XX pairs on neighbouring
    data qubits just before each measurement block, so one physical
    mechanism fires >2 detectors simultaneously.

    Data qubits identified from the before-round DEPOLARIZE1 target list.
    Neighbours approximated by consecutive qubit indices (synthetic but
    honest, labelled as such)."""
    text = str(build_circuit(distance, rounds, noise, memory))
    # data qubits = targets of the first before-round DEPOLARIZE1
    m = re.search(r"DEPOLARIZE1\([^)]+\)\s+([\d ]+)", text)
    dq = [int(q) for q in m.group(1).split()]
    pairs = list(zip(dq[::2], dq[1::2]))          # adjacent-index pairs
    lines = text.splitlines()
    out, n_ins = [], 0
    for ln in lines:
        if re.match(r"\s*M\s", ln):               # inject before each M block
            for a, b in pairs:
                out.append(f"        CORRELATED_ERROR({p_burst}) X{a} X{b}")
            n_ins += len(pairs)
        out.append(ln)
    print(f"burst injection: {n_ins} CORRELATED_ERROR XX pairs at p={p_burst} "
          f"across {len(pairs)} qubit pairs")
    return stim.Circuit("\n".join(out))


def sample(circuit, shots):
    sampler = circuit.compile_detector_sampler()
    det, obs = sampler.sample(shots, separate_observables=True)
    return det, obs[:, 0]


def mwpm(circuit, det, obs):
    dem = circuit.detector_error_model(decompose_errors=True)
    m = pymatching.Matching.from_detector_error_model(dem)
    pred = m.decode_batch(det.astype(bool))[:, 0]
    return float((pred != obs.astype(bool)).mean())


def calibrated_matching(nominal_circuit, det, validate=None):
    """Rebuild Matching with edge weights fitted from syndrome data.

    Interior edges: Spitz-style co-occurrence inversion.
    Boundary edges: residual marginals, the part of a detector's firing
    rate left unexplained by its fitted interior edges, shared across
    the detector's boundary edges.

    validate: optional (det_val, obs_val) pair. When given, decodes both
    nominal and calibrated matchings on the held-out shots and returns
    the better one. The honest guard against calibration hurting under
    noise the marginal model can't represent (measured: correlated
    bursts make fitted weights worse than nominal)."""
    dem = nominal_circuit.detector_error_model(decompose_errors=True)
    m_nom = pymatching.Matching.from_detector_error_model(dem)
    G = nx.Graph(m_nom.to_networkx())

    n_det = det.shape[1]
    marg = det.mean(axis=0)
    n_fit = 0
    fitted_p = {}                             # interior edge -> p_e
    for u, v, d in G.edges(data=True):
        if u >= n_det or v >= n_det:          # boundary virtual node
            continue
        mi, mj = marg[u], marg[v]
        pij = float(np.logical_and(det[:, u], det[:, v]).mean())
        denom = 1.0 - mi - mj + pij
        pe = (pij - mi * mj) / denom if denom > 1e-9 else mi * mj
        pe = float(np.clip(pe, 1e-6, 0.49))
        d["weight"] = float(np.log((1.0 - pe) / pe))
        fitted_p[(u, v)] = pe
        n_fit += 1

    # boundary edges: share each detector's residual rate among its
    # boundary mechanisms
    inc_interior = {i: 0.0 for i in range(n_det)}
    bnd_edges = {i: [] for i in range(n_det)}
    for u, v, d in G.edges(data=True):
        if u < n_det and v < n_det:
            inc_interior[u] += fitted_p[(u, v)]
            inc_interior[v] += fitted_p[(u, v)]
        elif u < n_det:
            bnd_edges[u].append((u, v))
        elif v < n_det:
            bnd_edges[v].append((u, v))
    n_bnd = 0
    for i, edges in bnd_edges.items():
        resid = float(np.clip(marg[i] - inc_interior[i], 1e-6, 0.49))
        for u, v in edges:
            pe = resid / len(edges)
            G.edges[u, v]["weight"] = float(np.log((1.0 - pe) / pe))
            n_bnd += 1
    print(f"calibration: fitted {n_fit} interior + {n_bnd} boundary edges")

    m_cal = pymatching.Matching(G)
    if validate is None:
        return m_cal
    det_v, obs_v = validate
    e_nom = float((m_nom.decode_batch(det_v.astype(bool))[:, 0]
                   != obs_v.astype(bool)).mean())
    e_cal = float((m_cal.decode_batch(det_v.astype(bool))[:, 0]
                   != obs_v.astype(bool)).mean())
    pick = m_cal if e_cal <= e_nom else m_nom
    print(f"validation: nominal {e_nom:.4f} vs calibrated {e_cal:.4f} "
          f"-> using {'calibrated' if e_cal <= e_nom else 'NOMINAL'}")
    return pick


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
    p.add_argument("--burst", type=float, default=0.0,
                   help="correlated XX fault probability (0 = off)")
    p.add_argument("--model", default=None, choices=["mlp", "cnn"],
                   help="also train a learned decoder")
    p.add_argument("--shots", type=int, default=200_000)
    args = p.parse_args()

    np.random.seed(SEED)
    nominal = build_circuit(args.distance, args.rounds, args.noise,
                            memory=args.memory)
    if args.burst > 0:
        truth = burst_circuit(args.distance, args.rounds, args.noise,
                              args.burst, memory=args.memory)
    elif args.eta == 1.0:
        truth = nominal
    else:
        truth = biased_circuit(args.distance, args.rounds, args.noise,
                               args.eta, memory=args.memory,
                               hetero=args.hetero)

    det, obs = sample(truth, args.shots)
    split = args.shots // 2
    # calibration shard halves: fit on first, validate on second
    det_fit, det_val = det[: split // 2], det[split // 2: split]
    obs_val = obs[split // 2: split]
    det_te, obs_te = det[split:], obs[split:]

    print(f"\nd={args.distance} r={args.rounds} p={args.noise} eta={args.eta} "
          f"| {split} calibration + {split} test shots\n")

    naive = mwpm(nominal, det_te, obs_te)
    m_cal = calibrated_matching(nominal, det_fit,
                                validate=(det_val, obs_val))
    pred_cal = m_cal.decode_batch(det_te.astype(bool))[:, 0]
    cal_err = float((pred_cal != obs_te.astype(bool)).mean())
    try:
        informed = mwpm(truth, det_te, obs_te)
    except Exception:
        informed = None  # non-graphlike DEM: no informed MWPM exists

    print(f"naive MWPM (depolarising prior):   {naive:.4f}")
    print(f"calibrated MWPM (fitted from data):{cal_err:>9.4f}")
    if informed is None:
        print("informed MWPM (oracle true DEM):   N/A (non-graphlike DEM, "
              "no matching representation)")
    else:
        print(f"informed MWPM (oracle true DEM):   {informed:.4f}")

    if args.model:
        from qec_decoder import MLPDecoder, ConvDecoder, train_decoder
        det_f = det.astype(np.float32)
        obs_f = obs.astype(np.float32)
        if args.model == "cnn":
            coords = {i: c for i, c in
                      truth.get_detector_coordinates().items()}
            net = ConvDecoder(coords)
        else:
            net = MLPDecoder(truth.num_detectors)
        err, t = train_decoder(net, det_f[:split], obs_f[:split],
                               det_f[split:], obs_f[split:], epochs=40)
        print(f"{args.model.upper()} learned decoder:          "
              f"{err:.4f}  ({t:.0f}s)")

    gap = (naive - cal_err)
    print(f"\ncalibrated-vs-naive delta: {gap:+.4f}")


if __name__ == "__main__":
    main()
