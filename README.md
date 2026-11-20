# qec-decoder

Learned quantum error-correction decoders vs classical baselines, on a
laptop. `stim` + `PyMatching` + `PyTorch`.

The question: under which noise conditions can a learned (or
data-calibrated) decoder beat minimum-weight perfect matching, the
classical decoder everyone uses?

## The problem

A logical qubit survives only if the decoder guesses the right errors:

```mermaid
flowchart LR
    E[physical errors<br/>decoherence, gate faults] --> S[stabiliser<br/>measurements]
    S --> D{{DECODER}}
    D -->|correct guess| OK[logical qubit<br/>survives]
    D -->|wrong guess| X[logical qubit<br/>corrupts]
    style D fill:#7c3aed,color:#fff
```

Every decoder decodes against an **assumed noise model**. On real
hardware the true noise is different: biased, correlated, drifting.

```mermaid
flowchart TB
    subgraph real[" reality "]
        TN[true noise<br/>biased + correlated] --> C[quantum circuit]
        C --> SY[syndrome data<br/>detection events]
    end
    subgraph assumed[" decoder's belief "]
        AM[assumed model<br/>i.i.d. depolarising] --> MW[MWPM<br/>edge weights]
    end
    SY --> MW
    MW --> DEC[correction]
    TN -.->|mismatch = decoding errors| MW
    SY -.->|"this project: refit weights<br/>from the data itself"| MW
    style real fill:#fef3c7
    style assumed fill:#f3efff
```

## What I measured, and where the room actually is

```mermaid
flowchart TD
    N[true noise] --> Q{edge-representable<br/>on the matching graph?}
    Q -->|depolarising| A[MWPM near-optimal<br/>learned decoder loses: ~12% vs 5.2%]
    Q -->|"biased / rate-miscalibrated<br/>(still edge-representable)"| B[MWPM indifferent<br/>oracle gains only ~2%]
    Q -->|"multi-body correlated<br/>leakage, crosstalk, bursts"| C["MWPM CAN'T represent it<br/>= the only regime a learned<br/>decoder can win →"]
    style A fill:#fee2e2
    style B fill:#fef3c7
    style C fill:#d1fae5
```

The first two rows are measured in this repo. The third is the next
experiment, and it is also why Google's AlphaQubit only showed its
advantage on real device noise.

## Scripts

| Script | What it does |
|---|---|
| `qec_decoder.py` | stim surface-code circuit → sampled syndromes → MWPM vs MLP/transformer logical error rates. Flags: `--distance --rounds --noise --model {mlp,transformer,both} --mismatch` |
| `calibrated_baseline.py` | Adds the honest classical adversary: MWPM re-weighted by fitting edge probabilities from syndrome co-occurrence (Spitz et al. 1712.02360), plus biased-noise injection via `PAULI_CHANNEL_1` and correlated `CORRELATED_ERROR` bursts. Flags: `--eta --hetero --burst --mlp --noise --distance --memory {x,z}` |
| `viz_decoder.py` | One-shot decode picture: fired defects by round, matched pairs, boundary matches, plus a live naive-vs-calibrated LER chart |
| `test_e2e.py` | End-to-end invariants: zero-noise sanity, depolarising sanity, biased-DEM validity, the robustness finding as a regression test |

## Measured findings (d=3-5, rotated memory-Z)

```
Decoder comparison, depolarising p=0.005, d=3, r=9:
  MWPM (calibrated):        5.2%
  Transformer decoder:     11.9%
  MLP decoder:             12.5%

Weight-miscalibration study (calibrated_baseline.py):
  MWPM, 4x wrong rate prior, d=3:          ~4.6%  (unchanged)
  Uniform X-bias eta=100, d=3:     naive 5.74% = oracle 5.74%
  Hetero X/Z-bias eta=100, d=3:    naive 4.57% = oracle 4.57%
  Hetero X/Z-bias eta=100, d=5:    naive 13.55%, oracle 13.29% (~2% rel)

Correlated XX bursts (p_burst=0.15, p_base=0.001, d=3, 3 seeds):
  naive MWPM:     3.04%  (boundary broken)
  MLP learned:    1.81%  (recovers ~60% of naive->oracle gap,
                          McNemar p<0.001 per seed)
  oracle MWPM:    1.00%
```

**Honest reading.** Under correctly-calibrated i.i.d. depolarising noise,
MWPM is near-optimal and a small learned decoder underperforms it. That
is the expected result, not a failure. More interesting: MWPM's logical
error rate is essentially insensitive to edge-weight miscalibration
whenever the true noise is still edge-representable (uniform AND
alternating bias). Weight fitting only has room to matter where noise is
not edge-representable: multi-body correlated faults. Measured both
sides of that boundary now: under correlated XX bursts a small MLP beats
naive MWPM by ~40% relative with McNemar p<0.001, recovering about 60%
of what the oracle gets back. See `DESIGN.md`.

## Run

```bash
uv venv .venv --python 3.11
uv pip install -r requirements.txt

.venv/bin/python qec_decoder.py --model both
.venv/bin/python calibrated_baseline.py --eta 1.0          # sanity
.venv/bin/python calibrated_baseline.py --eta 100 --hetero # structural mismatch
.venv/bin/python viz_decoder.py                          # picture + LER chart
.venv/bin/python test_e2e.py                             # 11 invariant checks
```

`one-pager.html` is a self-contained summary of the problem and findings.

## Roadmap

`DESIGN.md` has the noise suite (biased, drift, burst), calibrated-MWPM
baseline, ConvNet/hybrid decoders, threshold curves. License: Apache-2.0.
