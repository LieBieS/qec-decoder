"""Reproducibility pack: every number in the README/DESIGN re-derivable.

Generates the canonical sampled shards (detection events + observable
flips) for each measured regime, with the stim sampler seed recorded.
Numbers in docs should come from THESE shards, so anyone can re-run the
same decode comparisons byte-for-byte.

Run: .venv/bin/python repro_pack.py
"""

import json
import os

import numpy as np

from calibrated_baseline import (biased_circuit, build_circuit,
                                 burst_circuit)

OUT = "results/shards"

REGIMES = {
    # name: (circuit_builder, kwargs, seed)
    "depolarising_d3": (build_circuit,
                        dict(distance=3, rounds=9, noise=0.005), 11),
    "biased_hetero_d3": (biased_circuit,
                         dict(distance=3, rounds=9, noise=0.005,
                              eta=100.0, hetero=True), 12),
    "burst_d3": (burst_circuit,
                 dict(distance=3, rounds=9, noise=0.001,
                      p_burst=0.15), 13),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = {}
    for name, (builder, kw, seed) in REGIMES.items():
        circuit = builder(**kw)
        det, obs = circuit.compile_detector_sampler(seed=seed).sample(
            50_000, separate_observables=True)
        np.savez_compressed(
            f"{OUT}/{name}.npz",
            det=det, obs=obs[:, 0],
            det_coords=np.array([
                v for _, v in sorted(
                    circuit.get_detector_coordinates().items())]))
        manifest[name] = {
            "builder": builder.__name__, "kwargs": kw,
            "sampler_seed": seed, "shots": 50_000,
            "n_detectors": circuit.num_detectors,
        }
        print(f"{name}: {det.shape} saved")
    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest written to {OUT}/manifest.json")


if __name__ == "__main__":
    main()
