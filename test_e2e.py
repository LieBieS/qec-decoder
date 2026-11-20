"""End-to-end pipeline test — exercises the whole stack and asserts
invariants, not just "it ran."

  T1  zero-noise circuit -> every decoder returns ~0 logical errors
      (if this fails, the harness itself is broken)
  T2  depolarizing: calibrated MWPM ~= naive ~= informed
      (calibration must not hurt a correct prior)
  T3  biased circuit builds, DEM decomposes, LER is sane
  T4  biased-noise data -> calibrated MWPM within tolerance of naive
      (edge-representable noise => weight fitting is a no-op;
      the measured finding, encoded as a regression test)
  T5  MLP decoder trains and beats chance (LER < 0.5)
  T6  viz_decoder produces a PNG

Run: .venv/bin/python test_e2e.py
"""

import os
import subprocess
import sys

import numpy as np
import pymatching
import stim

from calibrated_baseline import (biased_circuit, build_circuit,
                                 calibrated_matching, mwpm, sample)

SHOTS = 30_000
passed = []


def check(name, ok, detail=""):
    passed.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")


print("T1: zero-noise invariants")
c = build_circuit(3, 3, 0.0)
det, obs = sample(c, 2000)
mwpm_ler = mwpm(c, det, obs)
check("MWPM zero LER", mwpm_ler == 0.0, f"({mwpm_ler})")
check("no observable flips", obs.mean() == 0.0)
check("no detection events", det.mean() == 0.0)
# NB: calibration is undefined at zero noise — empty DEM has no graph

print("\nT2: depolarizing sanity (p=0.005, d=3)")
nom = build_circuit(3, 9, 0.005)
det, obs = sample(nom, SHOTS)
te = slice(SHOTS // 2, None)
naive = mwpm(nom, det[te], obs[te])
m_cal = calibrated_matching(nom, det[: SHOTS // 2])
cal = float((m_cal.decode_batch(det[te].astype(bool))[:, 0]
             != obs[te].astype(bool)).mean())
check("naive MWPM sane", 0.01 < naive < 0.15, f"({naive:.4f})")
check("calibrated ~= naive", abs(cal - naive) < 0.01,
      f"({cal:.4f} vs {naive:.4f})")

print("\nT3: biased circuit builds + valid DEM")
bc = biased_circuit(3, 9, 0.005, 100.0, hetero=True)
dem_ok = False
try:
    dem = bc.detector_error_model(decompose_errors=True)
    dem_ok = pymatching.Matching.from_detector_error_model(dem) is not None
except Exception as e:
    print("   DEM error:", e)
check("biased DEM decomposes for PyMatching", dem_ok)

print("\nT4: edge-representable mismatch — calibration is a no-op "
      "(measured finding as regression test)")
det, obs = sample(bc, SHOTS)
naive_b = mwpm(nom, det[te], obs[te])
m_cal = calibrated_matching(nom, det[: SHOTS // 2])
cal_b = float((m_cal.decode_batch(det[te].astype(bool))[:, 0]
               != obs[te].astype(bool)).mean())
oracle_b = mwpm(bc, det[te], obs[te])
check("calibrated ~= naive under structural bias",
      abs(cal_b - naive_b) < 0.02, f"({cal_b:.4f} vs {naive_b:.4f})")
check("oracle ~= naive (the robustness result)",
      abs(oracle_b - naive_b) < 0.02, f"({oracle_b:.4f} vs {naive_b:.4f})")

print("\nT5: MLP decoder trains and beats chance")
from qec_decoder import MLPDecoder, train_decoder
det_f, obs_f = det.astype(np.float32), obs.astype(np.float32)
mlp = MLPDecoder(bc.num_detectors)
mlp_ler, t = train_decoder(mlp, det_f[: SHOTS // 2], obs_f[: SHOTS // 2],
                           det_f[te], obs_f[te], epochs=8)
check("MLP beats chance", mlp_ler < 0.5, f"({mlp_ler:.4f}, {t:.0f}s)")

print("\nT6: viz produces output")
r = subprocess.run([sys.executable, "viz_decoder.py",
                    "--eval-shots", "4000"],
                   capture_output=True, text=True)
check("viz_decoder exits clean", r.returncode == 0,
      r.stderr.strip()[-120:] if r.returncode else "")
check("PNG written", os.path.exists("results/decoder_viz.png")
      and os.path.getsize("results/decoder_viz.png") > 10_000)

print(f"\n{'=' * 44}\n{sum(passed)}/{len(passed)} checks passed")
sys.exit(0 if all(passed) else 1)
