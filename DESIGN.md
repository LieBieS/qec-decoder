# QEC Learned-Decoder Research System — Design

**Status:** architecture v1 · **Author:** lead dev · **Date:** 2026-11-19

---

## 1. Problem statement

Build a research system that trains and evaluates learned quantum
error-correction decoders against classical baselines across a controlled
suite of noise models, and answers one question rigorously:

> **Under which noise conditions does a learned decoder beat a calibrated
> classical decoder — and by how much?**

This is the question Google's AlphaQubit answered on real Sycamore data
(ML decoder > MWPM under real, correlated device noise). Our laptop-scale
version isolates the *mechanism*: the decoder's job is posterior estimation
p(logical error | syndrome), so a learned decoder wins exactly when the
classical decoder's internal noise model is misspecified — not when it's
merely miscalibrated in rate. (Measured this session: MWPM survives a 4×
rate miscalibration almost untouched; the graph structure is what matters.)

### Goals

- Config-driven experiments: noise model × decoder, all paired on identical
  sampled shots
- Publication-standard metrics: logical error rate with binomial CI,
  logical-vs-physical curves, threshold estimates
- A noise suite that covers the regimes where the literature says learned
  decoders win
- A hybrid decoder track: MWPM output as features into an ML corrector —
  the strongest known pattern in this literature

### Non-goals

- Real-time decoding latency constraints (hardware-loop problem)
- Fault-tolerance proofs, exact fidelity guarantees
- Competing with AlphaQubit's training budget — we work the mechanism,
  not the leaderboard

---

## 2. The core architectural decision

**Noise model is a first-class axis, not a parameter.**

The single load-bearing insight from this session's experiments: decoder
advantage is a property of the *mismatch* between the noise that generated
the data and the noise model the decoder assumes. So the system models
noise twice:

- `NoiseSpec` — what physically corrupts the circuit (the data generator)
- the decoder's prior — what the decoder believes the noise is
  (MWPM's DEM, the ML model's training distribution)

Every experiment is a (generator noise, decoder prior) pair. The diagonal
(matched) is the sanity row; the off-diagonal is where the science lives.

```
                    DECODER PRIOR
                  true      wrong-rate    wrong-structure
             ┌───────────┬────────────┬──────────────────┐
   GEN   dep │ E1 sanity │ E1b robust │                  │
   NOISE biased│          │            │ E2  ★ ML's game  │
        drift │          │            │ E3  ★ ML's game  │
        burst │          │            │ E4  ★ ML's game  │
             └───────────┴────────────┴──────────────────┘
```

---

## 3. System layers

```
qec/
  noise.py      NoiseSpec hierarchy + circuit builders (stim)
  data.py       syndrome dataset generation, disk cache, split policy
  decoders/
    base.py     Decoder interface
    mwpm.py     PyMatching wrapper (calibrated or specified DEM)
    mlp.py      dev-loop baseline
    convnet.py  2D+time CNN — strong spatial inductive bias
    transformer.py per-detector-token attention
    hybrid.py   MWPM features -> ML corrector
  eval/
    metrics.py  logical error rate, CI, McNemar paired test
    curves.py   p_L-vs-p_phys sweep + threshold estimate
  experiments/
    run.py      config -> results/experiments.csv (+ dataset registry)
    configs.py  named experiment specs
  tests/
    test_sanity.py  invariants (zero-noise -> zero logical error, etc.)
```

### 3.1 `noise.py` — the noise suite

| NoiseSpec | Construction (stim) | Why it's in the suite |
|---|---|---|
| `Depolarizing(p)` | `stim.Circuit.generated` params | sanity baseline; MWPM near-optimal |
| `Biased(p, eta, axis)` | custom circuit: error prob along `axis` = `p·η`, others `p`, normalized to fixed total rate | biased noise degrades standard MWPM — the classic ML-win regime. **Axis must oppose the tracked observable**: memory-Z tracks Z_L, so Z-bias is a rigged game (Z errors never flip it — ML wins trivially by learning "never flips"); use X-bias on memory-Z or Z-bias on memory-X |
| `Burst(p, len)` | injected correlated multi-qubit flips | correlated errors violate MWPM's independence assumption |
| `Drift(p0→p1)` | per-round noise ramp (unrolled REPEAT) | nonstationary — ML adapts to distribution, DEM can't |
| `MeasAsym(p0,p1)` | asymmetric measurement flip probs | simple wrong-structure case |

Each `NoiseSpec` exposes `build(distance, rounds) -> stim.Circuit` and
`decoder_dem()` — the DEM a *naive* decoder would use (always depolarizing
at nominal p), distinct from the true generating noise.

### 3.2 `data.py` — datasets as versioned artifacts

- `Sample(config_hash) -> .npz` cache: det events, observable flips,
  config metadata. Same shard feeds every decoder in a comparison —
  paired evaluation, not separate sampling.
- Split policy: train/test by shard, seeds fixed; `experiments.csv`
  records the dataset hash so runs are comparable across sessions.

### 3.3 `decoders/` — the interface

```python
class Decoder(Protocol):
    name: str
    def fit(self, det: np.ndarray, obs: np.ndarray) -> None: ...
    def predict(self, det: np.ndarray) -> np.ndarray: ...  # obs bits
```

- `MWPMDecoder(dem=spec.decoder_dem() | spec.true_dem())` — both modes,
  so "calibrated" and "naive" MWPM are explicit experiment rows.
- `MLPDecoder` — exists, keep as smoke-test model (trains in seconds).
- `ConvDecoder` — syndrome reshaped to (time, x, y) via detector
  coordinates; a surface-code syndrome is literally a 3D image, CNNs are
  the literature's strongest cheap model.
- `TransformerDecoder` — exists; keep for small models, but it's
  CPU-expensive — not the iteration vehicle.
- `HybridDecoder` — MWPM posterior (edge marginals / matched-pair features)
  concatenated with raw syndrome → small net. Best-known pattern;
  cheap to train, strong results in literature.

### 3.4 `eval/` — honesty machinery (carried from our rules)

- Logical error rate + binomial CI (95%), reported per config
- McNemar paired test on decoder disagreements (same shots)
- p_L(p_phys) curves over distance {3,5,7} → threshold estimate
- every row in experiments.csv: config hash, dataset hash, metrics, CI,
  train time, eval time, git sha

### 3.5 `tests/` — invariants

- zero-noise circuit → every decoder outputs zero logical error
- MWPM at tiny p recovers exact error on single-fault cases
- dataset cache keyed by config hash actually reproduces

---

## 4. Experiment matrix (the run order)

| ID | Noise | Decoders | Expected | Gate |
|---|---|---|---|---|
| E0 | Depolarizing d3 | all | MWPM wins — sanity | pipeline reproduces today's numbers |
| E0.5 | Biased η d3 | **data-calibrated MWPM** (DEM fit from sampled syndromes, not from spec) | recovers most of the gap — the honest classical adversary | MUST run before E1; without it every later claim is circular |

  E0.5 implementation (verified, fix-qec.md): estimate each DEM edge's
  error probability from pairwise defect co-occurrence — Spitz et al.
  arXiv:1712.02360 (weight = log((1−p)/p)); also Chen/IBM
  arXiv:2110.04285 and Blume-Kohout/Young arXiv:2504.14643. PyMatching
  accepts custom weights via `Matching.from_check_matrix(H, weights=…)`
  or NetworkX edges with `error_probability` attrs — rebuild the
  `Matching` per calibration (cheap at d≤7; no in-place reweighting,
  see open PyMatching issues #172/#173 — landing a maintained estimator
  is itself a contribution).
| E1 | Biased η, X-axis, d3 | naive-MWPM vs calibrated-MWPM vs MLP/CNN | ML win *only* counts vs calibrated-MWPM | gap > CI vs BOTH MWPM rows |
| E2 | Biased η, X-axis, d3 | informed-MWPM vs ML | MWPM recovers when told truth | bounds "value of knowing the noise" |
| E3 | Drift d3 | MWPM(nominal) vs ML on dist. | ML adapts | |
| E4 | Burst d3 | MWPM vs ML | ML wins on correlated faults | |
| E5 | d∈{3,5,7} best configs | threshold curves | the publication figure | |
| E6 | stretch: Google QEC dataset | loader + decoder | real-device comparison | |

E1+E2 together are the money experiment: they isolate *structural* model
error from decoder quality and produce a statement of the form "learned
decoders recover X% of the gap a correct noise model provides, without
being told the noise model." That statement is the finding.

---

## 5. Milestones

- **M0 — refactor to package + feature-shaped artifact** (half day):
  split qec_decoder.py into the layout above; E0 reproduces
  5.2%/12.5%/11.9% on identical shards. NOTE per adversarial review:
  grants (Deltakit Community Fund) pay for SDK *features* — error
  budgeting, threshold computation — not solo results. Shape at least one
  M0-M2 deliverable as an upstream-contributable feature (e.g., a DEM
  calibration-from-data utility or threshold-fit tool), since the proven
  career path is tools others run, not isolated results.
- **M0.5 — Deltakit fund application (URGENT, deadline Dec 2 2026)**:
  Riverlane×Unitary Community Fund has scoped open issues paying
  $2-4k/feature. Two are nearly this project's own deliverables:
  **#422 threshold estimation via finite-size scaling** and **#321
  decoder calibration scoring (Brier/reliability) using PyMatching soft
  output**. Application window Nov 2–Dec 2. Submit a proposal framed
  around landing one of these as an upstream Deltakit feature.
- **M1 — noise suite + biased circuit** (day): `Biased` custom circuit —
  NOTE the confirmed fix: stim has no bias parameter on generated
  circuits, use `PAULI_CHANNEL_1(px,py,pz)`/`PAULI_CHANNEL_2` for
  arbitrary bias; honest pairing = memory-X under Z-bias (or X-bias on
  memory-Z). XZZX variant optional (reference stim circuits exist at
  github.com/jetxezarreta/qec-two-level-qubits-circuit-noise-bias).
  Verified via stim DEM + pymatching accepting the decomposed model
- **M2 — E1/E2 runs** (day): first ML-beats-MWPM result under biased noise,
  with CI — or an honest negative written up
- **M3 — hybrid decoder + ConvNet** (2 days): strongest local model
- **M4 — E5 curves + writeup** (2 days): threshold figure, findings doc

## 6. Risks

| Risk | Mitigation |
|---|---|
| MWPM still wins under biased noise (it reweights edges correctly when the DEM is right) | that's E2's explicit hypothesis — the win is vs *naive* MWPM, which is the honest real-world condition |
| Custom biased circuit won't decompose to a valid DEM | verify `circuit.detector_error_model(decompose_errors=True)` early in M1; fall back to stim's built-in biased params if present |
| Transformer too slow on CPU | ConvNet is the primary learned model; transformer stays optional |
| Correlated-noise circuit realism | keep burst model clearly labeled synthetic; real noise comes from E6's Google dataset |

## 7. Standing rules (from the research discipline)

- Identical shards for every decoder in a comparison — paired stats only
- Tuned classical baseline is mandatory (informed-MWPM, not just naive)
- Structural ablation counts as evidence: decoder without spatial
  structure should underperform — if it doesn't, the structure wasn't load-bearing
- Negatives get written up with the same care as wins
