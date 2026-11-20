# qec-decoder

Learned quantum error-correction decoders vs classical baselines, on a
laptop. `stim` + `PyMatching` + `PyTorch`.

## What it does now

`qec_decoder.py` runs an end-to-end experiment:

- stim generates a rotated surface-code memory-Z experiment
  (`--distance`, `--rounds`, `--noise`)
- samples detection events + true observable flips
- decodes with MWPM (PyMatching) — the classical baseline
- trains an MLP or transformer decoder on the same shots
- reports logical error rate for each

Reference numbers (d=3, r=9, p=0.005, depolarizing):

```
MWPM (calibrated):        5.2%
Transformer decoder:     11.9%
MLP decoder:             12.5%
MWPM (4x miscalibrated):  4.6%   <- surprisingly robust
```

Honest reading: under correctly-calibrated i.i.d. depolarizing noise MWPM
is near-optimal and a small learned decoder underperforms it — parity or
better is *not* the expectation. The regimes where learned decoders earn
their keep are structural noise mismatch (bias, correlation, drift) —
that's what the roadmap targets.

## Run

```bash
uv venv .venv --python 3.11
uv pip install -r requirements.txt
.venv/bin/python qec_decoder.py --model both
# simulate a miscalibrated MWPM:
.venv/bin/python qec_decoder.py --model mlp --mismatch 0.02
```

## Roadmap

See `DESIGN.md` — noise-suite (`PAULI_CHANNEL_1` biased noise, drift,
burst), a data-calibrated MWPM baseline (Spitz-style edge-weight fitting
from syndrome statistics), ConvNet/hybrid decoders, and threshold curves.
