
import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pampfl_full import (
    init_encoder, forward, bce, fed_avg, sgd_step, DPSGDPrivacy,
    run_classical_baselines,  # kept importable for comparison if you want it
)

RNG = np.random.default_rng(42)


# ------------------------------------------------------------------
# 1. Feature extraction per patient (mirrors the literature's approach
#    of 24 time/frequency-domain features over full-day / day / night
#    segments — see e.g. the Depresjon Random-Forest studies)
# ------------------------------------------------------------------
def _segment_stats(values):
    if len(values) < 3:
        return [0.0] * 6
    return [
        float(np.mean(values)), float(np.std(values)), float(np.median(values)),
        float(np.max(values)), float(skew(values)), float(kurtosis(values)),
    ]


def extract_patient_features(csv_path):
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    hour = df["timestamp"].dt.hour
    day_mask = (hour >= 8) & (hour < 20)

    full = df["activity"].to_numpy(dtype=float)
    day = df.loc[day_mask, "activity"].to_numpy(dtype=float)
    night = df.loc[~day_mask, "activity"].to_numpy(dtype=float)

    feats = _segment_stats(full) + _segment_stats(day) + _segment_stats(night)
    feats.append(float(np.mean(full == 0)) if len(full) else 0.0)  # sedentary fraction
    feats.append(float(len(full)))                                  # recording length
    return np.array(feats, dtype=float)


def _find_dir(root, name):
    """condition/ and control/ might be nested one level deeper depending
    on how the zip was extracted — search a couple of levels for them."""
    for dirpath, dirnames, _ in os.walk(root):
        if name in dirnames:
            return os.path.join(dirpath, name)
    raise FileNotFoundError(f"Could not find a '{name}' folder under {root}")


def load_depresjon(data_dir):
    X, y, patient_ids = [], [], []
    for label, folder_name in [(1, "condition"), (0, "control")]:
        folder = _find_dir(data_dir, folder_name)
        for csv_path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
            X.append(extract_patient_features(csv_path))
            y.append(label)
            patient_ids.append(os.path.splitext(os.path.basename(csv_path))[0])
    X, y = np.vstack(X), np.array(y)
    print(f"Loaded {len(X)} patients ({y.sum()} condition / {(1 - y).sum()} control), "
          f"{X.shape[1]} features each.")
    return X, y, patient_ids


# ------------------------------------------------------------------
# 2. Partition patients into simulated institutional clients
#    (Section 6.2: several "hospital" clients, one held out for
#    cross-site validation)
# ------------------------------------------------------------------
def partition_into_clients(X, y, patient_ids, n_clients=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    splits = np.array_split(idx, n_clients)
    return [{"X": X[s], "y": y[s], "ids": [patient_ids[i] for i in s]} for s in splits]


# ------------------------------------------------------------------
# 3. Run PAMP-FL's horizontal FedAvg + DP-SGD on the real data,
#    reusing the exact same functions already validated in pampfl_full.py
# ------------------------------------------------------------------
def run_on_depresjon(data_dir, n_clients=5, rounds=15):
    X, y, patient_ids = load_depresjon(data_dir)
    X = StandardScaler().fit_transform(X)  # z-score features before training

    clients = partition_into_clients(X, y, patient_ids, n_clients=n_clients)
    holdout = clients.pop()  # one simulated hospital held out for cross-site validation

    global_model = init_encoder(X.shape[1])
    privacy = DPSGDPrivacy(clip=1.0, noise_std=0.05)

    print(f"\n[PAMP-FL on Depresjon] {len(clients)} training clients "
          f"({[len(c['X']) for c in clients]} patients each) + 1 held-out site "
          f"({len(holdout['X'])} patients)\n")

    history = {"round": [], "acc": [], "loss": [], "f1": []}
    for r in range(1, rounds + 1):
        updates = []
        for c in clients:
            local = privacy.train({k: v.copy() for k, v in global_model.items()},
                                   c["X"], c["y"], epochs=5, lr=0.1, batch_size=8)
            updates.append((local, len(c["X"])))
        global_model = fed_avg(updates)

        _, p = forward(global_model, holdout["X"])
        pred = (p.flatten() > 0.5).astype(int)
        acc = accuracy_score(holdout["y"], pred)
        f1 = f1_score(holdout["y"], pred, zero_division=0)
        loss = bce(p, holdout["y"])
        history["round"].append(r)
        history["acc"].append(float(acc))
        history["loss"].append(float(loss))
        history["f1"].append(float(f1))
        print(f"  Round {r:2d}: held-out accuracy={acc:.3f}  F1={f1:.3f}  loss={loss:.4f}")

    _, p = forward(global_model, holdout["X"])
    pred = (p.flatten() > 0.5).astype(int)
    print("\nFinal held-out-site report:")
    print(f"  Accuracy : {accuracy_score(holdout['y'], pred):.3f}")
    print(f"  Precision: {precision_score(holdout['y'], pred, zero_division=0):.3f}")
    print(f"  Recall   : {recall_score(holdout['y'], pred, zero_division=0):.3f}")
    print(f"  F1-score : {f1_score(holdout['y'], pred, zero_division=0):.3f}")

    return global_model, history


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "depresjon"
    if not os.path.isdir(data_dir):
        print(f"Data directory '{data_dir}' not found.\n"
              f"Download+unzip the Depresjon  first (see the module docstring), "
              f"then run: python3 depresjon_pipeline.py /path/to/depresjon")
        sys.exit(1)
    run_on_depresjon(data_dir)


"""""
#output :
#Loaded 55 patients (23 condition / 32 control), 20 features each.

[PAMP-FL on Depresjon] 4 training clients ([11, 11, 11, 11] patients each) + 1 held-out site (11 patients)

  Round  1: held-out accuracy=0.545  F1=0.545  loss=0.6662
  Round  2: held-out accuracy=0.727  F1=0.727  loss=0.6530
  Round  3: held-out accuracy=0.727  F1=0.727  loss=0.6270
  Round  4: held-out accuracy=0.727  F1=0.727  loss=0.6114
  Round  5: held-out accuracy=0.818  F1=0.833  loss=0.5904
  Round  6: held-out accuracy=0.818  F1=0.833  loss=0.6108
  Round  7: held-out accuracy=0.727  F1=0.727  loss=0.6170
  Round  8: held-out accuracy=0.818  F1=0.833  loss=0.6060
  Round  9: held-out accuracy=0.818  F1=0.857  loss=0.6035
  Round 10: held-out accuracy=0.727  F1=0.769  loss=0.6298
  Round 11: held-out accuracy=0.727  F1=0.769  loss=0.6402
  Round 12: held-out accuracy=0.818  F1=0.857  loss=0.6303
  Round 13: held-out accuracy=0.727  F1=0.769  loss=0.6373
  Round 14: held-out accuracy=0.727  F1=0.769  loss=0.6681
 # Round 15: held-out accuracy=0.727  F1=0.769  loss=0.6671

#Final held-out-site report:
 # Accuracy : 0.727
 # Precision: 0.833
 # Recall   : 0.714
 # F1-score : 0.769
 """"