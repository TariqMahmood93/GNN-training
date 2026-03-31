"""
GNN Node Classification Pipeline — German Credit Dataset
Phases: Setup → Training → Validation
"""

import os
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "hidden_dim": 64,
    "dropout": 0.5,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "epochs": 200,
    "k_neighbors": 5,
}

DATASET_DIR = "datasets"
DATA_FILE = os.path.join(DATASET_DIR, "german.data")
MODEL_FILE = os.path.join(DATASET_DIR, "trained_model.pt")
LOSS_CURVE_FILE = os.path.join(DATASET_DIR, "training_loss_curve.png")
CONFUSION_MATRIX_FILE = os.path.join(DATASET_DIR, "validation_confusion_matrix.png")

# German Credit Dataset column names (per UCI documentation)
COLUMN_NAMES = [
    "checking_account",      # A1  — categorical
    "duration",              # A2  — numerical
    "credit_history",        # A3  — categorical
    "purpose",               # A4  — categorical
    "credit_amount",         # A5  — numerical
    "savings_account",       # A6  — categorical
    "employment",            # A7  — categorical
    "installment_rate",      # A8  — numerical
    "personal_status",       # A9  — categorical
    "other_debtors",         # A10 — categorical
    "residence_since",       # A11 — numerical
    "property",              # A12 — categorical
    "age",                   # A13 — numerical
    "other_installments",    # A14 — categorical
    "housing",               # A15 — categorical
    "existing_credits",      # A16 — numerical
    "job",                   # A17 — categorical
    "num_dependents",        # A18 — numerical
    "telephone",             # A19 — categorical
    "foreign_worker",        # A20 — categorical
    "target",                # A21 — 1=good, 2=bad
]

NUMERICAL_COLS = [
    "duration", "credit_amount", "installment_rate",
    "residence_since", "age", "existing_credits", "num_dependents",
]

# ---------------------------------------------------------------------------
# Step 1 — Dataset Acquisition
# ---------------------------------------------------------------------------

def _generate_synthetic_german_credit(path: str) -> None:
    """
    Generate a synthetic dataset that mirrors the German Credit schema
    (1000 rows, same column types and target distribution ~70/30).
    Used as a fallback when the UCI repository is unreachable.
    """
    rng = np.random.default_rng(42)
    n = 1000

    # Categorical columns — use real UCI attribute codes (e.g. A11, A32, A143)
    cat_codes: dict[str, list[str]] = {
        "checking_account":  ["A11", "A12", "A13", "A14"],
        "credit_history":    ["A30", "A31", "A32", "A33", "A34"],
        "purpose":           ["A40", "A41", "A42", "A43", "A44",
                              "A45", "A46", "A47", "A48", "A49", "A410"],
        "savings_account":   ["A61", "A62", "A63", "A64", "A65"],
        "employment":        ["A71", "A72", "A73", "A74", "A75"],
        "personal_status":   ["A91", "A92", "A93", "A94", "A95"],
        "other_debtors":     ["A101", "A102", "A103"],
        "property":          ["A121", "A122", "A123", "A124"],
        "other_installments":["A141", "A142", "A143"],
        "housing":           ["A151", "A152", "A153"],
        "job":               ["A171", "A172", "A173", "A174"],
        "telephone":         ["A191", "A192"],
        "foreign_worker":    ["A201", "A202"],
    }
    data: dict[str, np.ndarray] = {}
    for col, codes in cat_codes.items():
        data[col] = rng.choice(codes, size=n)

    # Numerical columns — approximate UCI distributions
    data["duration"]         = rng.integers(4, 72, size=n)
    data["credit_amount"]    = rng.integers(250, 18500, size=n)
    data["installment_rate"] = rng.integers(1, 4, size=n)
    data["residence_since"]  = rng.integers(1, 4, size=n)
    data["age"]              = rng.integers(19, 75, size=n)
    data["existing_credits"] = rng.integers(1, 4, size=n)
    data["num_dependents"]   = rng.integers(1, 2, size=n)

    # Target: ~70% good (1), ~30% bad (2) — matches real dataset distribution
    data["target"] = rng.choice([1, 2], size=n, p=[0.70, 0.30])

    df = pd.DataFrame(data, columns=COLUMN_NAMES)
    df.to_csv(path, sep=" ", index=False, header=False)
    print(f"  Synthetic dataset written to {path}")


def download_data() -> None:
    """
    Download the German Credit Dataset from UCI.
    Falls back to a synthetic equivalent if the network is unavailable.
    """
    os.makedirs(DATASET_DIR, exist_ok=True)
    if os.path.exists(DATA_FILE):
        print(f"  Data file already exists: {DATA_FILE}")
        return
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/statlog/german/german.data"
    )
    print(f"  Downloading dataset from {url} …")
    try:
        urllib.request.urlretrieve(url, DATA_FILE)
        print(f"  Saved to {DATA_FILE}")
    except Exception as exc:
        print(f"  Download failed ({exc}).")
        print("  Falling back to synthetic dataset with identical schema …")
        _generate_synthetic_german_credit(DATA_FILE)


# ---------------------------------------------------------------------------
# Step 2 — Preprocessing
# ---------------------------------------------------------------------------

def preprocess() -> tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess the German Credit dataset.

    Returns
    -------
    X : np.ndarray  shape (n_samples, n_features)  — normalised feature matrix
    y : np.ndarray  shape (n_samples,)              — binary labels (0=good, 1=bad)
    """
    df = pd.read_csv(DATA_FILE, sep=" ", header=None, names=COLUMN_NAMES)

    # Remap target: 1 → 0 (good credit), 2 → 1 (bad credit)
    df["target"] = df["target"].map({1: 0, 2: 1})

    y = df["target"].values.astype(np.int64)
    df = df.drop(columns=["target"])

    # Encode categorical columns with LabelEncoder
    categorical_cols = [c for c in df.columns if c not in NUMERICAL_COLS]
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Normalise numerical features
    scaler = StandardScaler()
    df[NUMERICAL_COLS] = scaler.fit_transform(df[NUMERICAL_COLS])

    X = df.values.astype(np.float32)

    # Print dataset statistics
    n_samples, n_features = X.shape
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"  Samples   : {n_samples}")
    print(f"  Features  : {n_features}")
    print(f"  Class dist: 0 (good)={class_dist.get(0, 0)}, "
          f"1 (bad)={class_dist.get(1, 0)}")

    return X, y


# ---------------------------------------------------------------------------
# Step 3 — Graph Construction + Node Masks
# ---------------------------------------------------------------------------

def build_graph(X: np.ndarray, y: np.ndarray) -> Data:
    """
    Build a KNN graph (k=5) from the tabular feature matrix and wrap it
    into a PyTorch Geometric Data object. Also attaches train/val/test masks.
    """
    k = CONFIG["k_neighbors"]
    print(f"  Building KNN graph with k={k} …")

    # Build KNN adjacency (directed; convert to COO edge_index)
    A = kneighbors_graph(X, n_neighbors=k, mode="connectivity", include_self=False)
    cx = A.tocoo()
    edge_index = torch.tensor(
        np.vstack([cx.row, cx.col]), dtype=torch.long
    )

    x_tensor = torch.tensor(X, dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.long)

    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)

    # ---- Node splits -------------------------------------------------------
    n = x_tensor.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))

    n_train = int(0.60 * n)
    n_val   = int(0.20 * n)

    train_idx = perm[:n_train]
    val_idx   = perm[n_train : n_train + n_val]
    test_idx  = perm[n_train + n_val :]

    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.val_mask   = torch.zeros(n, dtype=torch.bool)
    data.test_mask  = torch.zeros(n, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx]     = True
    data.test_mask[test_idx]   = True

    print(f"  Nodes: {n} | Edges: {edge_index.size(1)}")
    print(f"  Train: {data.train_mask.sum().item()} | "
          f"Val: {data.val_mask.sum().item()} | "
          f"Test: {data.test_mask.sum().item()}")

    return data


# ---------------------------------------------------------------------------
# Step 4 — Model Definition
# ---------------------------------------------------------------------------

class GCN(torch.nn.Module):
    """Two-layer Graph Convolutional Network for node classification."""

    def __init__(self, in_channels: int, hidden_dim: int,
                 out_channels: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def build_model(in_channels: int, num_classes: int = 2) -> GCN:
    return GCN(
        in_channels=in_channels,
        hidden_dim=CONFIG["hidden_dim"],
        out_channels=num_classes,
        dropout=CONFIG["dropout"],
    )


# ---------------------------------------------------------------------------
# Phase 1 — Training
# ---------------------------------------------------------------------------

def train(data: Data) -> None:
    """Train the GCN and save weights + loss curve."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = build_model(in_channels=data.num_node_features).to(device)
    data  = data.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    criterion = torch.nn.CrossEntropyLoss()

    loss_history: list[float] = []

    model.train()
    for epoch in range(1, CONFIG["epochs"] + 1):
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{CONFIG['epochs']} — Loss: {loss.item():.4f}")

    # Save model checkpoint
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"  Model saved to {MODEL_FILE}")

    # Save loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, CONFIG["epochs"] + 1), loss_history, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training Loss Curve — GCN (German Credit)")
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_FILE, dpi=150)
    plt.close()
    print(f"  Loss curve saved to {LOSS_CURVE_FILE}")


# ---------------------------------------------------------------------------
# Phase 2 — Validation
# ---------------------------------------------------------------------------

def validate(data: Data) -> None:
    """Load saved checkpoint and evaluate on the validation set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load fresh model from checkpoint
    model = build_model(in_channels=data.num_node_features).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    print(f"  Loaded weights from {MODEL_FILE}")

    data = data.to(device)
    model.eval()

    with torch.no_grad():
        out   = model(data.x, data.edge_index)
        preds = out.argmax(dim=1)

    val_preds  = preds[data.val_mask].cpu().numpy()
    val_labels = data.y[data.val_mask].cpu().numpy()

    # Classification report
    print("\n  Classification Report (Validation Set):")
    report = classification_report(
        val_labels, val_preds,
        target_names=["Good Credit (0)", "Bad Credit (1)"],
        digits=4,
    )
    print(report)

    # Confusion matrix plot
    cm = confusion_matrix(val_labels, val_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    classes = ["Good (0)", "Bad (1)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix — Validation Set")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {CONFUSION_MATRIX_FILE}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    # ---- Setup ----------------------------------------------------------------
    print("=== Setup: Dataset Acquisition ===")
    download_data()
    X, y = preprocess()
    data = build_graph(X, y)

    # ---- Phase 1 --------------------------------------------------------------
    print("\n=== Phase 1: Training ===")
    train(data)

    # ---- Phase 2 --------------------------------------------------------------
    print("\n=== Phase 2: Validation ===")
    validate(data)


if __name__ == "__main__":
    main()
