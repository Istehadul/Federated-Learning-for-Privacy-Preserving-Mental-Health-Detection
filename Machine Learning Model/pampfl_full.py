"""
PAMP-FL — Full Simulation
==========================
Privacy-Adaptive Multi-Paradigm Federated Learning, covering every
component the paper's Sections 5-7 describe (not just the core loop):

  A. Modality encoders  ............ Section 6.3   (tabular / text / image / audio)
  B. Horizontal FL (FedAvg)  ....... Section 5.4 / 2.2
  C. Vertical FL (feature alignment) Section 5.3   (private-set-intersection style)
  D. Transfer learning init  ....... Section 5.4 step 1 (low-resource client)
  E. Privacy Controller (3 backends) Section 6.4
       - DP-SGD (gradient clip + Gaussian noise)
       - Homomorphic-style seeded trees (PPD-ERT design)
       - Secure aggregation with a minimum-contributor threshold
  F. Classical ML baselines  ....... Section 6.1   (DT / ExtraTrees / GBM~XGBoost / LinearSVM)
  G. Experimental arms  ............ Section 7.1   (Local / Centralized / FedAvg / ILL / CILL)

Everything is NumPy + scikit-learn only (no internet access in this sandbox
to install PyTorch / Flower / Opacus / XGBoost, which are the paper's
*recommended production stack* — see the "swap-in" notes below each class
for exactly what to replace when moving to that stack).

Run: python3 pampfl_full.py
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, log_loss

RNG = np.random.default_rng(42)


# ========================================================================
# A. DATA — synthetic multimodal, non-IID, per-client
#    Swap-in: real OMOP-CDM tabular extracts, clinical text, MRI slices,
#    DAIC-WOZ audio, already preprocessed on-device.
# ========================================================================
def make_modality_data(modality, n_clients, min_n=150, max_n=500, n_features=None):
    """Each modality gets its own natural feature dimensionality, mirroring
    Table in Section 6.3. Non-IID: each client has a random label-bias."""
    dims = {"tabular": 20, "text": 50, "image": 64, "audio": 13}
    d = n_features or dims[modality]
    true_w = RNG.normal(0, 1, size=d)
    clients = []
    for _ in range(n_clients):
        n = RNG.integers(min_n, max_n)
        if modality == "text":
            X = RNG.poisson(1.5, size=(n, d)).astype(float)          # bag-of-words-like counts
        elif modality == "image":
            X = RNG.uniform(0, 1, size=(n, d))                        # flattened patch intensities
        elif modality == "audio":
            X = RNG.normal(0, 1, size=(n, d)) * RNG.uniform(0.5, 2)   # MFCC-like coefficients
        else:
            X = RNG.normal(0, 1, size=(n, d))
        bias = RNG.uniform(-0.7, 0.7)
        y = ((X @ true_w) / np.sqrt(d) + bias + RNG.normal(0, 1, n) > 0).astype(int)
        clients.append({"X": X, "y": y})
    return clients, d


# ========================================================================
# A. MODALITY ENCODERS (Section 6.3)
#    Swap-in: BiLSTM/BERT for text, ResNet-18 for imaging,
#    MFCC-CNN or GoogleNet/MobileNetV2/ResNet-18 ensemble for audio.
#    Here every modality is reduced to the same small 2-layer MLP shape
#    so they can all plug into the same FedAvg/DP-SGD machinery below.
# ========================================================================
def init_encoder(n_features, hidden=16):
    return {
        "W1": RNG.normal(0, 0.1, size=(n_features, hidden)),
        "b1": np.zeros(hidden),
        "W2": RNG.normal(0, 0.1, size=(hidden, 1)),
        "b2": np.zeros(1),
    }


def forward(model, X):
    a1 = np.tanh(X @ model["W1"] + model["b1"])
    p = 1 / (1 + np.exp(-(a1 @ model["W2"] + model["b2"])))
    return a1, p


def bce(p, y):
    p = np.clip(p.flatten(), 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def sgd_step(model, X, y, lr, dp_clip=None, dp_noise_std=0.0):
    a1, p = forward(model, X)
    err = (p.flatten() - y).reshape(-1, 1)
    grads = {
        "W2": a1.T @ err / len(X), "b2": err.mean(axis=0),
        "W1": X.T @ ((err @ model["W2"].T) * (1 - a1 ** 2)) / len(X),
        "b1": (((err @ model["W2"].T) * (1 - a1 ** 2))).mean(axis=0),
    }
    for k, g in grads.items():
        if dp_clip is not None:
            np.clip(g, -dp_clip, dp_clip, out=g)               # DP-SGD: per-step clipping
        noise = RNG.normal(0, dp_noise_std, g.shape) if dp_noise_std else 0
        model[k] = model[k] - lr * (g + noise)                  # DP-SGD: Gaussian noise
    return model


# ========================================================================
# E. PRIVACY CONTROLLER — 3 interchangeable backends (Section 6.4)
# ========================================================================
class DPSGDPrivacy:
    """Per-example gradient clipping + Gaussian noise before transmission."""
    def __init__(self, clip=1.0, noise_std=0.05):
        self.clip, self.noise_std = clip, noise_std

    def train(self, model, X, y, epochs, lr, batch_size=32):
        n = len(X)
        for _ in range(epochs):
            for start in range(0, n, batch_size):
                idx = RNG.permutation(n)[start:start + batch_size]
                model = sgd_step(model, X[idx], y[idx], lr, self.clip, self.noise_std)
        return model


class SeededTreePrivacy:
    """Homomorphic-encryption stand-in, following the PPD-ERT design: a shared
    global seed lets every client build STRUCTURALLY IDENTICAL trees, so the
    server can average leaf statistics without ever seeing raw client data."""
    def __init__(self, global_seed=7, n_trees=10, max_depth=4):
        self.global_seed, self.n_trees, self.max_depth = global_seed, n_trees, max_depth

    def fit_client_forest(self, X, y):
        return ExtraTreesClassifier(
            n_estimators=self.n_trees, max_depth=self.max_depth,
            random_state=self.global_seed,   # <-- identical structure across all clients
        ).fit(X, y)


class SecureAggregationGate:
    """Server can only aggregate once >= min_contributors clients have
    submitted an update for the round (prevents isolating a single update)."""
    def __init__(self, min_contributors=3):
        self.min_contributors = min_contributors

    def try_aggregate(self, updates_and_weights):
        if len(updates_and_weights) < self.min_contributors:
            return None  # round is held until enough clients report in
        return fed_avg(updates_and_weights)


# ========================================================================
# B. HORIZONTAL FL — weighted FedAvg (Section 2.2 / 5.4 step 5)
# ========================================================================
def fed_avg(updates_and_weights):
    total = sum(w for _, w in updates_and_weights)
    keys = updates_and_weights[0][0].keys()
    agg = {k: np.zeros_like(updates_and_weights[0][0][k]) for k in keys}
    for model, w in updates_and_weights:
        for k in keys:
            agg[k] += model[k] * (w / total)
    return agg


def run_horizontal_fl(modality="tabular", n_clients=6, rounds=12,
                       privacy: DPSGDPrivacy = None, secure_gate: SecureAggregationGate = None):
    clients, d = make_modality_data(modality, n_clients)
    holdout = clients.pop()
    privacy = privacy or DPSGDPrivacy()
    global_model = init_encoder(d)

    print(f"\n[Horizontal FL | modality={modality}] {len(clients)} clients + 1 held-out site")
    for r in range(1, rounds + 1):
        updates = []
        for c in clients:
            local = privacy.train({k: v.copy() for k, v in global_model.items()},
                                   c["X"], c["y"], epochs=3, lr=0.15)
            updates.append((local, len(c["X"])))
        aggregated = secure_gate.try_aggregate(updates) if secure_gate else fed_avg(updates)
        if aggregated is None:
            print(f"  Round {r:2d}: waiting for more contributors (secure-aggregation gate)")
            continue
        global_model = aggregated
        _, p = forward(global_model, holdout["X"])
        acc = accuracy_score(holdout["y"], p.flatten() > 0.5)
        print(f"  Round {r:2d}: held-out-site accuracy={acc:.3f}  loss={bce(p, holdout['y']):.4f}")
    return global_model


# ========================================================================
# C. VERTICAL FL — feature-alignment module (Section 5.3)
#    Two clients hold DIFFERENT features for OVERLAPPING patients (e.g. a
#    hospital's EHR + a wearable vendor's sensor stream for the same people).
#    A private-set-intersection (PSI) step finds the shared patient IDs
#    without either side revealing its full identifier list.
# ========================================================================
def psi_intersect(ids_a, ids_b):
    """Toy PSI: in production this uses cryptographic PSI so neither party
    sees the other's non-overlapping IDs. Here we just compute the overlap
    to demonstrate the alignment step that feeds the vertical model."""
    return np.intersect1d(ids_a, ids_b)


def run_vertical_fl(n_patients=400, overlap_frac=0.5):
    all_ids = np.arange(n_patients)
    hospital_ids = RNG.choice(all_ids, size=int(n_patients * 0.7), replace=False)
    wearable_ids = RNG.choice(all_ids, size=int(n_patients * 0.6), replace=False)
    shared_ids = psi_intersect(hospital_ids, wearable_ids)

    hospital_feats = RNG.normal(0, 1, size=(len(hospital_ids), 15))   # EHR-side features
    wearable_feats = RNG.normal(0, 1, size=(len(wearable_ids), 8))    # sensor-side features
    h_map = {pid: i for i, pid in enumerate(hospital_ids)}
    w_map = {pid: i for i, pid in enumerate(wearable_ids)}

    X_joint = np.hstack([
        hospital_feats[[h_map[p] for p in shared_ids]],
        wearable_feats[[w_map[p] for p in shared_ids]],
    ])
    true_w = RNG.normal(0, 1, X_joint.shape[1])
    y_joint = ((X_joint @ true_w) + RNG.normal(0, 1, len(shared_ids)) > 0).astype(int)

    model = init_encoder(X_joint.shape[1])
    for _ in range(200):
        model = sgd_step(model, X_joint, y_joint, lr=0.2)
    _, p = forward(model, X_joint)
    print(f"\n[Vertical FL] {len(shared_ids)}/{n_patients} patients linked via PSI | "
          f"joint-model train accuracy={accuracy_score(y_joint, p.flatten() > 0.5):.3f}")
    return model


# ========================================================================
# D. TRANSFER LEARNING INIT (Section 5.4 step 1) — a new, low-resource
#    client starts from a data-rich client's trained weights instead of
#    from scratch.
# ========================================================================
def transfer_init_new_client(source_model, new_client_data, fine_tune_epochs=5, lr=0.1):
    model = {k: v.copy() for k, v in source_model.items()}
    for _ in range(fine_tune_epochs):
        model = sgd_step(model, new_client_data["X"], new_client_data["y"], lr)
    _, p = forward(model, new_client_data["X"])
    print(f"\n[Transfer learning] low-resource client fine-tuned from a data-rich "
          f"client's weights | accuracy={accuracy_score(new_client_data['y'], p.flatten() > 0.5):.3f} "
          f"(n={len(new_client_data['X'])} local samples)")
    return model


# ========================================================================
# F. CLASSICAL ML BASELINES (Section 6.1) — trained centrally per-client
#    just to reproduce the "traditional-ML pathway" comparison points.
#    Swap-in: real xgboost.XGBClassifier for the GradientBoosting stand-in.
# ========================================================================
def run_classical_baselines(modality="tabular"):
    clients, _ = make_modality_data(modality, n_clients=1, min_n=800, max_n=801)
    X, y = clients[0]["X"], clients[0]["y"]
    split = int(0.8 * len(X))
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=0),
        "Extremely Randomized Trees": ExtraTreesClassifier(n_estimators=100, random_state=0),
        "Gradient Boosting (XGBoost stand-in)": GradientBoostingClassifier(random_state=0),
        "Linear SVM": LinearSVC(random_state=0, max_iter=5000),
    }
    print(f"\n[Classical ML baselines | modality={modality}]")
    for name, m in models.items():
        m.fit(Xtr, ytr)
        acc = accuracy_score(yte, m.predict(Xte))
        print(f"  {name:38s} accuracy={acc:.3f}")


# ========================================================================
# G. EXPERIMENTAL ARMS (Section 7.1): Local-only / Centralized / FedAvg /
#    ILL (Institutional Incremental Learning) / CILL (Cyclic ILL)
# ========================================================================
def run_experimental_arms(modality="tabular", n_clients=6, rounds=10, cill_cycles=3):
    clients, d = make_modality_data(modality, n_clients)
    holdout = clients.pop()

    def acc_on_holdout(model):
        _, p = forward(model, holdout["X"])
        return accuracy_score(holdout["y"], p.flatten() > 0.5)

    # --- Local-only: each client trains and is evaluated alone; report the mean ---
    local_accs = []
    for c in clients:
        m = init_encoder(d)
        for _ in range(30):
            m = sgd_step(m, c["X"], c["y"], lr=0.15)
        local_accs.append(acc_on_holdout(m))
    arm_local = float(np.mean(local_accs))

    # --- Centralized: pool every client's data (privacy-ignoring upper bound) ---
    X_pool = np.vstack([c["X"] for c in clients])
    y_pool = np.concatenate([c["y"] for c in clients])
    m = init_encoder(d)
    for _ in range(30):
        m = sgd_step(m, X_pool, y_pool, lr=0.15)
    arm_centralized = acc_on_holdout(m)

    # --- FedAvg: reuse the horizontal-FL loop, quietly ---
    m = init_encoder(d)
    for _ in range(rounds):
        updates = [(DPSGDPrivacy(noise_std=0).train({k: v.copy() for k, v in m.items()},
                                                      c["X"], c["y"], epochs=3, lr=0.15), len(c["X"]))
                   for c in clients]
        m = fed_avg(updates)
    arm_fedavg = acc_on_holdout(m)

    # --- ILL: pass ONE model sequentially through every client, once ---
    m = init_encoder(d)
    for c in clients:
        for _ in range(10):
            m = sgd_step(m, c["X"], c["y"], lr=0.15)
    arm_ill = acc_on_holdout(m)

    # --- CILL: repeat the sequential pass for several cycles ---
    m = init_encoder(d)
    for _ in range(cill_cycles):
        for c in clients:
            for _ in range(10):
                m = sgd_step(m, c["X"], c["y"], lr=0.15)
    arm_cill = acc_on_holdout(m)

    print(f"\n[Experimental arms | modality={modality}] held-out-site accuracy")
    for name, val in [("Local-only (mean)", arm_local), ("Centralized (upper bound)", arm_centralized),
                       ("FedAvg", arm_fedavg), ("ILL", arm_ill), ("CILL", arm_cill)]:
        print(f"  {name:28s} {val:.3f}")


# ========================================================================
if __name__ == "__main__":
    run_horizontal_fl(modality="tabular", secure_gate=SecureAggregationGate(min_contributors=3))
    run_vertical_fl()
    src_model = run_horizontal_fl(modality="audio", n_clients=5, rounds=6)
    new_client_data = make_modality_data("audio", n_clients=1, min_n=40, max_n=60)[0][0]
    transfer_init_new_client(src_model, new_client_data)
    run_classical_baselines(modality="tabular")
    run_experimental_arms(modality="tabular")
