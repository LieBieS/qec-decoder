# QEC Learned-Decoder Research System, Design

**Status:** architecture v1 · **Author:** Shaun · **Date:** 2026-11-19

---

## 1. Problem statement

Build a research system that trains and evaluates learned quantum
error-correction decoders against classical baselines across a controlled
suite of noise models, and answers one question rigorously:

> **Under which noise conditions does a learned decoder beat a calibrated
> classical decoder, and by how much?**

This is the question Google's AlphaQubit answered on real Sycamore data
(ML decoder > MWPM under real, correlated device noise). Our laptop-scale
version isolates the mechanism: the decoder's job is posterior estimation
p(logical error | syndrome), so a learned decoder wins exactly when the
classical decoder's internal noise model is misspecified, not when it's
merely miscalibrated in rate. (Measured this session: MWPM survives a 4x
rate miscalibration almost untouched; the graph structure is what
matters.)

### Goals

- Config-driven experiments: noise model x decoder, all paired on
  identical sampled shots
- Publication-standard metrics: logical error rate with binomial CI,
  logical-vs-physical curves, threshold estimates
- A noise suite that covers the regimes where the literature says learned
  decoders win
- A hybrid decoder track: MWPM output as features into an ML corrector,
  the strongest known pattern in this literature

### Non-goals

- Real-time decoding latency constraints (hardware-loop problem)
- Fault-tolerance proofs, exact fidelity guarantees
- Competing with AlphaQubit's training budget: we work the mechanism,
  not the leaderboard

---

## 2. The core architectural decision

**Noise model is a first-class axis, not a parameter.**

The single load-bearing insight from this session's experiments: decoder
advantage is a property of the mismatch between the noise that generated
the data and the noise model the decoder assumes. So the system models
noise twice:

- `NoiseSpec`, what physically corrupts the circuit (the data generator)
- the decoder's prior, what the decoder believes the noise is
  (MWPM's DEM, the ML model's training distribution)

Every experiment is a (generator noise, decoder prior) pair. The diagonal
(matched) is the sanity row; the off-diagonal is where the science lives.

```
                    DECODER PRIOR
                  true      wrong-rate    wrong-structure
             +-----------+------------+------------------+
   GEN   dep | E1 sanity | E1b robust |                  |
   NOISE biased|         |            | E2  * ML's game  |
        drift |          |            | E3  * ML's game  |
        burst |          |            | E4  * ML's game  |
             +-----------+------------+------------------+
```

Measured update (this session): the "wrong-rate" column is empirically
dead. Uniform and alternating bias at eta=100 left MWPM's LER unchanged
at d=3-5, and the oracle recovered only ~2%. Any noise that still maps
onto the matching graph's edges gives MWPM nothing to lose. The live
column is wrong-structure only: correlated, multi-body faults.

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
    convnet.py  2D+time CNN, strong spatial inductive bias
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

### 3.1 `noise.py`, the noise suite

| NoiseSpec | Construction (stim) | Why it's in the suite |
|---|---|---|
| `Depolarising(p)` | `stim.Circuit.generated` params | sanity baseline; MWPM near-optimal |
| `Biased(p, eta, axis)` | `PAULI_CHANNEL_1` text-replace on the generated circuit (verified working in `calibrated_baseline.py`) | biased noise was the classic ML-win regime. **Axis must oppose the tracked observable**: memory-Z tracks Z_L, so Z-bias is a rigged game (Z errors never flip it, ML wins trivially by learning "never flips"); use X-bias on memory-Z or Z-bias on memory-X |
| `Burst(p, len)` | injected correlated multi-qubit flips | correlated errors violate MWPM's independence assumption. THE open experiment |
| `Drift(p0->p1)` | per-round noise ramp (unrolled REPEAT) | nonstationary: ML adapts to distribution, DEM can't |
| `MeasAsym(p0,p1)` | asymmetric measurement flip probs | simple wrong-structure case |

Each `NoiseSpec` exposes `build(distance, rounds) -> stim.Circuit` and
`decoder_dem()`, the DEM a naive decoder would use (always depolarising
at nominal p), distinct from the true generating noise.

### 3.2 `data.py`, datasets as versioned artefacts

- `Sample(config_hash) -> .npz` cache: det events, observable flips,
  config metadata. Same shard feeds every decoder in a comparison, so
  evaluation is paired, not separate sampling.
- Split policy: train/test by shard, seeds fixed; `experiments.csv`
  records the dataset hash so runs are comparable across sessions.

### 3.3 `decoders/`, the interface

```python
class Decoder(Protocol):
    name: str
    def fit(self, det: np.ndarray, obs: np.ndarray) -> None: ...
    def predict(self, det: np.ndarray) -> np.ndarray: ...  # obs bits
```

- `MWPMDecoder(dem=spec.decoder_dem() | spec.true_dem())`, both modes,
  so "calibrated" and "naive" MWPM are explicit experiment rows.
- `MLPDecoder`, exists, keep as smoke-test model (trains in seconds).
- `ConvDecoder`, syndrome reshaped to (time, x, y) via detector
  coordinates; a surface-code syndrome is literally a 3D image, CNNs are
  the literature's strongest cheap model.
- `TransformerDecoder`, exists; keep for small models, but it's
  CPU-expensive, not the iteration vehicle.
- `HybridDecoder`, MWPM posterior (edge marginals / matched-pair
  features) concatenated with raw syndrome into a small net. Best-known
  pattern; cheap to train, strong results in literature.

### 3.4 `eval/`, honesty machinery (carried from our rules)

- Logical error rate + binomial CI (95%), reported per config
- McNemar paired test on decoder disagreements (same shots)
- p_L(p_phys) curves over distance {3,5,7} then threshold estimate
- every row in experiments.csv: config hash, dataset hash, metrics, CI,
  train time, eval time, git sha

### 3.5 `tests/`, invariants

- zero-noise circuit gives zero logical error from every decoder
- MWPM at tiny p recovers exact error on single-fault cases
- dataset cache keyed by config hash actually reproduces

(Current `test_e2e.py` implements all three plus the robustness finding
as a regression check.)

---

## 4. Experiment matrix (the run order)

| ID | Noise | Decoders | Expected | Gate |
|---|---|---|---|---|
| E0 | Depolarising d3 | all | MWPM wins, sanity | pipeline reproduces today's numbers. DONE |
| E0.5 | Biased eta d3 | data-calibrated MWPM | honest classical adversary | DONE: `calibrated_baseline.py`. Finding: edge-representable noise leaves MWPM untouched |
| E1 | Biased eta, X-axis, d3 | naive vs calibrated vs MLP/CNN | ML win only counts vs calibrated-MWPM | gap > CI vs BOTH MWPM rows. Note: our E0.5 result predicts a null here unless the noise is non-edge-representable |
| E2 | Biased eta, X-axis, d3 | informed-MWPM vs ML | MWPM recovers when told truth | bounds "value of knowing the noise" |
| E3 | Drift d3 | MWPM(nominal) vs ML on dist. | ML adapts | |
| E4 | Burst d3 | MWPM vs ML | ML wins on correlated faults | THE critical remaining experiment |
| E5 | d in {3,5,7} best configs | threshold curves | the publication figure | also the promised #422 dataset |
| E6 | stretch: Google QEC dataset | loader + decoder | real-device comparison | |

The money experiment moved: E1+E2 measured a boundary, not a gap. E4
(burst) is now where a learned decoder can actually win, because
correlated multi-body faults are the only thing the matching graph
cannot represent. A clean E4 win vs calibrated MWPM is the result worth
publishing.

---

## 5. Milestones

- **M0, refactor to package + feature-shaped artefact** (half day):
  split into the layout above; E0 reproduces 5.2%/12.5%/11.9% on
  identical shards. Partly overtaken: scripts grew around the pipeline,
  refactor only if it earns its keep.
- **M0.5, Deltakit Community Fund (deadline 2 Dec 2026)**: own issue
  FILED as #435 (data-driven decoder calibration); engaged by #422's
  author (muf148) proposing a shared data format, per-round logical
  error with std devs across distances, for his threshold estimator.
  Promised deliverable: d=3,5,7 biased-noise LER comparison posted to
  #422. Remaining: record the 2-min video, submit the form.
- **M1, noise suite** (day): `Biased` DONE (PAULI_CHANNEL_1 replace,
  verified). `Burst` next: multi-qubit correlated flips, e.g.
  PAULI_CHANNEL_2 pairs on neighbouring data qubits or explicit
  correlated X/Y chains. `Drift` via unrolled REPEAT ramp.
- **M2, E4/E3 runs** (day): first ML-beats-MWPM result under
  non-edge-representable noise, with CI, or an honest negative
  written up.
- **M3, hybrid decoder + ConvNet** (2 days): strongest local model.
- **M4, E5 curves + writeup** (2 days): threshold figure, findings doc.
  E5 doubles as the promised #422 dataset.

## 6. Risks

| Risk | Mitigation |
|---|---|
| MWPM still wins under biased noise (it reweights edges correctly when the DEM is right) | MEASURED: it does. The win condition moved to E4's correlated faults, which no edge weights can capture |
| Learned decoder wins on burst but it's a rigged pairing (burst axis flips observable) | pair burst axis against the tracked observable, same rule as bias |
| Transformer too slow on CPU | ConvNet is the primary learned model; transformer stays optional |
| Correlated-noise circuit realism | keep burst model clearly labelled synthetic; real noise comes from E6's Google dataset |
| My pairwise calibrator is crude (NNLS-free, single-pass) | already measured: slightly worse than naive at p>=0.02. Document as limitation; Spitz's iterative refinement is the upgrade if it matters |

## 7. Standing rules (the research discipline)

- Identical shards for every decoder in a comparison, paired stats only
- Tuned classical baseline is mandatory (informed-MWPM, not just naive)
- Structural ablation counts as evidence: a decoder without spatial
  structure should underperform. If it doesn't, the structure wasn't
  load-bearing
- Negatives get written up with the same care as wins
