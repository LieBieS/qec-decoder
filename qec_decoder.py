"""ML decoder vs MWPM on the rotated surface code.

The question: can a neural network learn to decode error syndromes
(detection events) better than minimum-weight perfect matching?

- stim generates a rotated surface code memory-Z experiment with
  depolarising noise and samples detection events + the true observable flip
- pymatching decodes with MWPM -> classical baseline logical error rate
- torch models (MLP, tiny transformer) decode -> ML logical error rate

Interesting flag: --mismatch makes MWPM decode with a detector error model
calibrated at the wrong noise rate. On real hardware the noise model is
always miscalibrated; an ML decoder just learns the actual noise. This is
where neural decoders genuinely earn their keep.

Run: .venv/bin/python qec_decoder.py [--distance 3] [--noise 0.005]
"""

import argparse
import time

import numpy as np
import pymatching
import stim
import torch
import torch.nn as nn

SEED = 42


def build_circuit(distance, rounds, noise):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise,
        before_round_data_depolarization=noise,
        before_measure_flip_probability=noise,
        after_reset_flip_probability=noise,
    )


def sample(circuit, shots):
    sampler = circuit.compile_detector_sampler()
    det, obs = sampler.sample(shots, separate_observables=True)
    return det.astype(np.float32), obs[:, 0].astype(np.float32)


def mwpm_error_rate(circuit, det, obs):
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    pred = matching.decode_batch(det.astype(bool))[:, 0]
    return float((pred != obs.astype(bool)).mean())


# --- ML decoders -------------------------------------------------------------

class MLPDecoder(nn.Module):
    def __init__(self, n_det):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_det, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TransformerDecoder(nn.Module):
    """One token per detector; learned per-detector positional embedding
    encodes its (x, y, t) identity."""

    def __init__(self, n_det, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.det_embed = nn.Embedding(n_det, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        tok = self.embed(x.unsqueeze(-1)) + self.det_embed.weight
        out = self.encoder(tok)
        return self.head(out.mean(dim=1)).squeeze(-1)


class ConvDecoder(nn.Module):
    """Syndrome as a space-time image: each detector bit scattered to its
    (t, y, x) grid cell, time as input channels. The spatial inductive
    bias the MLP lacks; built for the correlated-noise regime where
    locality structure is the signal."""

    def __init__(self, coords):
        super().__init__()
        # coords: dict det_id -> (x, y, t)
        self.T = int(max(c[2] for c in coords.values())) + 1
        self.H = int(max(c[1] for c in coords.values())) + 1
        self.W = int(max(c[0] for c in coords.values())) + 1
        idx = torch.zeros(len(coords), 3, dtype=torch.long)
        for i, c in coords.items():
            idx[i] = torch.tensor([int(c[2]), int(c[1]), int(c[0])])
        self.register_buffer("idx", idx)
        self.conv = nn.Sequential(
            nn.Conv2d(self.T, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 8, 1), nn.ReLU())
        self.head = nn.Linear(8 * self.H * self.W, 1)

    def forward(self, x):
        grid = torch.zeros(x.shape[0], self.T, self.H, self.W,
                           device=x.device)
        grid[:, self.idx[:, 0], self.idx[:, 1], self.idx[:, 2]] = x
        return self.head(self.conv(grid).flatten(1)).squeeze(-1)


def train_decoder(model, X_tr, y_tr, X_te, y_te, epochs=30, bs=512, lr=3e-4):
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    tr = torch.utils.data.TensorDataset(
        torch.tensor(X_tr), torch.tensor(y_tr))
    dl = torch.utils.data.DataLoader(tr, batch_size=bs, shuffle=True)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        if (ep + 1) % 5 == 0:
            print(f"    epoch {ep + 1:>2}/{epochs}  loss={tot / len(tr):.4f}")
    model.eval()
    with torch.no_grad():
        pred = (model(torch.tensor(X_te)) > 0).numpy()
    return float((pred != y_te.astype(bool)).mean()), time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--distance", type=int, default=3)
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--noise", type=float, default=0.005)
    p.add_argument("--train-shots", type=int, default=50_000)
    p.add_argument("--test-shots", type=int, default=10_000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--model", default="transformer",
                   choices=["mlp", "transformer", "both"])
    p.add_argument("--mismatch", type=float, default=None,
                   help="calibrate MWPM at this p instead of the true noise")
    args = p.parse_args()

    np.random.seed(SEED)
    circuit = build_circuit(args.distance, args.rounds, args.noise)
    print(f"rotated memory-Z: d={args.distance} r={args.rounds} "
          f"p={args.noise}  detectors={circuit.num_detectors} "
          f"qubits={circuit.num_qubits}\n")

    print("sampling shots (stim)...")
    det_tr, obs_tr = sample(circuit, args.train_shots)
    det_te, obs_te = sample(circuit, args.test_shots)

    # MWPM baseline, optionally miscalibrated
    dem_circuit = (circuit if args.mismatch is None else
                   build_circuit(args.distance, args.rounds, args.mismatch))
    mwpm_err = mwpm_error_rate(dem_circuit, det_te, obs_te)
    tag = "" if args.mismatch is None else f" (calibrated p={args.mismatch})"
    print(f"MWPM logical error rate{tag}: {mwpm_err:.4f}\n")

    models = {"mlp": MLPDecoder, "transformer": TransformerDecoder}
    for name in ([args.model] if args.model != "both" else ["mlp", "transformer"]):
        print(f"{name} decoder:")
        model = models[name](circuit.num_detectors)
        err, t = train_decoder(model, det_tr, obs_tr, det_te, obs_te,
                               epochs=args.epochs)
        print(f"  {name} logical error rate: {err:.4f}  ({t:.0f}s)\n")

    print("note: with correctly-calibrated depolarising noise MWPM is "
          "near-optimal, parity is the honest result. Try --mismatch "
          "0.01 to simulate real-hardware miscalibration.")


if __name__ == "__main__":
    main()
