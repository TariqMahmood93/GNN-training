"""
GNN Node Classification Pipeline — German Credit Dataset
Phases: Setup → Training → Validation

Datasets are organised into per-dataset sub-folders:
  datasets/german_credit/   — German Credit (UCI Statlog)
  datasets/adult_census/    — Adult Census Income (UCI / US Census)
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

# Per-dataset directories
GERMAN_DIR = os.path.join("datasets", "german_credit")
ADULT_DIR  = os.path.join("datasets", "adult_census")

# German Credit paths
GERMAN_DATA_FILE          = os.path.join(GERMAN_DIR, "german.data")
GERMAN_CSV_FILE           = os.path.join(GERMAN_DIR, "german.csv")

# Pipeline output paths (German Credit)
MODEL_FILE            = os.path.join(GERMAN_DIR, "trained_model.pt")
LOSS_CURVE_FILE       = os.path.join(GERMAN_DIR, "training_loss_curve.png")
CONFUSION_MATRIX_FILE = os.path.join(GERMAN_DIR, "validation_confusion_matrix.png")

# Adult Census paths
ADULT_DATA_FILE = os.path.join(ADULT_DIR, "adult.data")
ADULT_CSV_FILE  = os.path.join(ADULT_DIR, "adult.csv")

# ---------------------------------------------------------------------------
# German Credit — schema
# ---------------------------------------------------------------------------

GERMAN_COLUMNS = [
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

GERMAN_NUMERICAL_COLS = [
    "duration", "credit_amount", "installment_rate",
    "residence_since", "age", "existing_credits", "num_dependents",
]

# ---------------------------------------------------------------------------
# Adult Census — schema
# ---------------------------------------------------------------------------

ADULT_COLUMNS = [
    "age",              # numerical
    "workclass",        # categorical
    "fnlwgt",           # numerical  (final sampling weight)
    "education",        # categorical
    "education_num",    # numerical
    "marital_status",   # categorical
    "occupation",       # categorical
    "relationship",     # categorical
    "race",             # categorical
    "sex",              # categorical
    "capital_gain",     # numerical
    "capital_loss",     # numerical
    "hours_per_week",   # numerical
    "native_country",   # categorical
    "income",           # target: <=50K / >50K
]

ADULT_NUMERICAL_COLS = [
    "age", "fnlwgt", "education_num",
    "capital_gain", "capital_loss", "hours_per_week",
]

ADULT_CAT_VALUES: dict[str, list[str]] = {
    "workclass": [
        "Private", "Self-emp-not-inc", "Self-emp-inc",
        "Federal-gov", "Local-gov", "State-gov",
        "Without-pay", "Never-worked",
    ],
    "education": [
        "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
        "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
        "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool",
    ],
    "marital_status": [
        "Married-civ-spouse", "Divorced", "Never-married",
        "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse",
    ],
    "occupation": [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces",
    ],
    "relationship": [
        "Wife", "Own-child", "Husband",
        "Not-in-family", "Other-relative", "Unmarried",
    ],
    "race": [
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black",
    ],
    "sex": ["Male", "Female"],
    "native_country": [
        "United-States", "Cuba", "Jamaica", "India", "Mexico",
        "South", "Japan", "Philippines", "Germany", "Canada",
        "Puerto-Rico", "El-Salvador", "Greece", "Italy", "China",
        "South-Korea", "Iran", "Poland", "Columbia", "Haiti",
        "Portugal", "Taiwan", "Hungary", "Honduras", "Ecuador",
        "Peru", "France", "Guatemala", "Nicaragua", "Vietnam",
        "Trinadad&Tobago", "Laos", "Thailand", "Yugoslavia",
        "Dominican-Republic", "Cambodia", "Ireland", "Hong", "Scotland",
        "Outlying-US(Guam-USVI-etc)", "England", "Holland-Netherlands",
    ],
}

# ===========================================================================
# ① German Credit Dataset
# ===========================================================================

def _generate_synthetic_german_credit(path: str) -> None:
    """Synthetic fallback using the real UCI categorical attribute codes."""
    rng = np.random.default_rng(42)
    n = 1000

    cat_codes: dict[str, list[str]] = {
        "checking_account":   ["A11", "A12", "A13", "A14"],
        "credit_history":     ["A30", "A31", "A32", "A33", "A34"],
        "purpose":            ["A40", "A41", "A42", "A43", "A44",
                               "A45", "A46", "A47", "A48", "A49", "A410"],
        "savings_account":    ["A61", "A62", "A63", "A64", "A65"],
        "employment":         ["A71", "A72", "A73", "A74", "A75"],
        "personal_status":    ["A91", "A92", "A93", "A94", "A95"],
        "other_debtors":      ["A101", "A102", "A103"],
        "property":           ["A121", "A122", "A123", "A124"],
        "other_installments": ["A141", "A142", "A143"],
        "housing":            ["A151", "A152", "A153"],
        "job":                ["A171", "A172", "A173", "A174"],
        "telephone":          ["A191", "A192"],
        "foreign_worker":     ["A201", "A202"],
    }

    data: dict[str, object] = {}
    for col, codes in cat_codes.items():
        data[col] = rng.choice(codes, size=n)

    data["duration"]         = rng.integers(4, 72, size=n)
    data["credit_amount"]    = rng.integers(250, 18500, size=n)
    data["installment_rate"] = rng.integers(1, 4, size=n)
    data["residence_since"]  = rng.integers(1, 4, size=n)
    data["age"]              = rng.integers(19, 75, size=n)
    data["existing_credits"] = rng.integers(1, 4, size=n)
    data["num_dependents"]   = rng.integers(1, 2, size=n)
    data["target"]           = rng.choice([1, 2], size=n, p=[0.70, 0.30])

    df = pd.DataFrame(data, columns=GERMAN_COLUMNS)
    df.to_csv(path, sep=" ", index=False, header=False)
    print(f"  Synthetic German Credit data written to {path}")


def download_german_data() -> None:
    """Download German Credit from UCI; fall back to synthetic if unavailable."""
    os.makedirs(GERMAN_DIR, exist_ok=True)

    if os.path.exists(GERMAN_DATA_FILE):
        print(f"  Already exists: {GERMAN_DATA_FILE}")
    else:
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases"
            "/statlog/german/german.data"
        )
        print(f"  Downloading German Credit from {url} …")
        try:
            urllib.request.urlretrieve(url, GERMAN_DATA_FILE)
            print(f"  Saved to {GERMAN_DATA_FILE}")
        except Exception as exc:
            print(f"  Download failed ({exc}). Using synthetic fallback …")
            _generate_synthetic_german_credit(GERMAN_DATA_FILE)

    # Always regenerate the CSV with headers
    df = pd.read_csv(
        GERMAN_DATA_FILE, sep=" ", header=None, names=GERMAN_COLUMNS
    )
    df.to_csv(GERMAN_CSV_FILE, index=False)
    print(f"  CSV saved to {GERMAN_CSV_FILE}")


def preprocess_german() -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess the German Credit dataset."""
    df = pd.read_csv(GERMAN_DATA_FILE, sep=" ", header=None, names=GERMAN_COLUMNS)

    # Remap target: 1 → 0 (good), 2 → 1 (bad)
    df["target"] = df["target"].map({1: 0, 2: 1})
    y = df.pop("target").values.astype(np.int64)

    categorical_cols = [c for c in df.columns if c not in GERMAN_NUMERICAL_COLS]
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df[GERMAN_NUMERICAL_COLS] = StandardScaler().fit_transform(df[GERMAN_NUMERICAL_COLS])
    X = df.values.astype(np.float32)

    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"  Samples  : {X.shape[0]} | Features: {X.shape[1]}")
    print(f"  Classes  : 0 (good)={dist.get(0,0)}, 1 (bad)={dist.get(1,0)}")
    return X, y


# ===========================================================================
# ② Adult Census Income Dataset
# ===========================================================================

def _generate_synthetic_adult(path: str) -> None:
    """
    Synthetic Adult Census dataset (48 842 rows matching real UCI distribution).
    Categorical columns use the exact string values from the original dataset.
    Target: <=50K (~75 %) / >50K (~25 %).
    """
    rng = np.random.default_rng(0)
    n = 48_842

    data: dict[str, object] = {}

    # Categorical columns with realistic weighted distributions
    workclass_w = [0.69, 0.08, 0.03, 0.03, 0.06, 0.04, 0.005, 0.005]
    data["workclass"] = rng.choice(
        ADULT_CAT_VALUES["workclass"], size=n,
        p=np.array(workclass_w) / sum(workclass_w),
    )

    education_w = [0.164, 0.223, 0.039, 0.316, 0.018, 0.033,
                   0.042, 0.016, 0.020, 0.014, 0.053, 0.006,
                   0.015, 0.013, 0.009, 0.002]
    data["education"] = rng.choice(
        ADULT_CAT_VALUES["education"], size=n,
        p=np.array(education_w) / sum(education_w),
    )

    marital_w = [0.459, 0.135, 0.328, 0.032, 0.033, 0.010, 0.003]
    data["marital_status"] = rng.choice(
        ADULT_CAT_VALUES["marital_status"], size=n,
        p=np.array(marital_w) / sum(marital_w),
    )

    occ_w = [0.029, 0.123, 0.107, 0.076, 0.099, 0.126, 0.042,
             0.062, 0.114, 0.031, 0.049, 0.015, 0.020, 0.003]
    data["occupation"] = rng.choice(
        ADULT_CAT_VALUES["occupation"], size=n,
        p=np.array(occ_w) / sum(occ_w),
    )

    rel_w = [0.105, 0.188, 0.401, 0.155, 0.032, 0.119]
    data["relationship"] = rng.choice(
        ADULT_CAT_VALUES["relationship"], size=n,
        p=np.array(rel_w) / sum(rel_w),
    )

    race_w = [0.855, 0.031, 0.010, 0.008, 0.096]
    data["race"] = rng.choice(
        ADULT_CAT_VALUES["race"], size=n,
        p=np.array(race_w) / sum(race_w),
    )

    data["sex"] = rng.choice(["Male", "Female"], size=n, p=[0.669, 0.331])

    # native_country: ~90 % United-States, rest spread across others
    other_countries = ADULT_CAT_VALUES["native_country"][1:]
    nc_p_other = np.ones(len(other_countries)) / len(other_countries) * 0.10
    nc_p = np.concatenate([[0.90], nc_p_other])
    data["native_country"] = rng.choice(
        ADULT_CAT_VALUES["native_country"], size=n, p=nc_p
    )

    # Numerical columns — approximate real UCI distributions
    data["age"]           = np.clip(rng.normal(38.6, 13.6, n).astype(int), 17, 90)
    data["fnlwgt"]        = rng.integers(13_000, 1_500_000, size=n)
    data["education_num"] = np.clip(rng.normal(10.1, 2.6, n).astype(int), 1, 16)
    data["capital_gain"]  = np.where(
        rng.random(n) < 0.92, 0,
        rng.integers(1, 99_999, size=n)
    )
    data["capital_loss"]  = np.where(
        rng.random(n) < 0.95, 0,
        rng.integers(1, 4_356, size=n)
    )
    data["hours_per_week"] = np.clip(rng.normal(40.4, 12.3, n).astype(int), 1, 99)

    # Target: ~75 % <=50K, ~25 % >50K (mirrors real dataset)
    data["income"] = rng.choice(["<=50K", ">50K"], size=n, p=[0.7592, 0.2408])

    df = pd.DataFrame(data, columns=ADULT_COLUMNS)
    # Write space-separated without header (matches UCI .data format)
    df.to_csv(path, index=False, header=False, sep=",")
    print(f"  Synthetic Adult Census data written to {path}  ({n} rows)")


def download_adult_data() -> None:
    """Download Adult Census from UCI; fall back to synthetic if unavailable."""
    os.makedirs(ADULT_DIR, exist_ok=True)

    if os.path.exists(ADULT_DATA_FILE):
        print(f"  Already exists: {ADULT_DATA_FILE}")
    else:
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases"
            "/adult/adult.data"
        )
        print(f"  Downloading Adult Census from {url} …")
        try:
            urllib.request.urlretrieve(url, ADULT_DATA_FILE)
            print(f"  Saved to {ADULT_DATA_FILE}")
        except Exception as exc:
            print(f"  Download failed ({exc}). Using synthetic fallback …")
            _generate_synthetic_adult(ADULT_DATA_FILE)

    # Regenerate CSV with headers
    df = pd.read_csv(
        ADULT_DATA_FILE, header=None, names=ADULT_COLUMNS,
        skipinitialspace=True, na_values="?",
    )
    df.dropna(inplace=True)
    df.to_csv(ADULT_CSV_FILE, index=False)
    print(f"  CSV saved to {ADULT_CSV_FILE}")

    # Print stats
    n_samples  = len(df)
    n_features = len(ADULT_COLUMNS) - 1
    dist = df["income"].value_counts().to_dict()
    print(f"  Samples  : {n_samples} | Features: {n_features}")
    print(f"  Classes  : <=50K={dist.get('<=50K', 0)}, >50K={dist.get('>50K', 0)}")


# ===========================================================================
# Shared — graph + model + training + validation
# ===========================================================================

def build_graph(X: np.ndarray, y: np.ndarray) -> Data:
    """Build a KNN graph (k=5) and return a PyG Data object with node masks."""
    k = CONFIG["k_neighbors"]
    print(f"  Building KNN graph with k={k} …")

    A  = kneighbors_graph(X, n_neighbors=k, mode="connectivity", include_self=False)
    cx = A.tocoo()
    edge_index = torch.tensor(np.vstack([cx.row, cx.col]), dtype=torch.long)

    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
    )

    n     = data.num_nodes
    perm  = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    n_tr  = int(0.60 * n)
    n_val = int(0.20 * n)

    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.val_mask   = torch.zeros(n, dtype=torch.bool)
    data.test_mask  = torch.zeros(n, dtype=torch.bool)
    data.train_mask[perm[:n_tr]]              = True
    data.val_mask  [perm[n_tr:n_tr + n_val]]  = True
    data.test_mask [perm[n_tr + n_val:]]      = True

    print(f"  Nodes: {n} | Edges: {edge_index.size(1)}")
    print(f"  Train: {data.train_mask.sum().item()} | "
          f"Val: {data.val_mask.sum().item()} | "
          f"Test: {data.test_mask.sum().item()}")
    return data


class GCN(torch.nn.Module):
    """Two-layer Graph Convolutional Network for node classification."""

    def __init__(self, in_channels: int, hidden_dim: int,
                 out_channels: int, dropout: float) -> None:
        super().__init__()
        self.conv1   = GCNConv(in_channels, hidden_dim)
        self.conv2   = GCNConv(hidden_dim, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def build_model(in_channels: int, num_classes: int = 2) -> GCN:
    return GCN(
        in_channels=in_channels,
        hidden_dim=CONFIG["hidden_dim"],
        out_channels=num_classes,
        dropout=CONFIG["dropout"],
    )


def train(data: Data, model_file: str, loss_curve_file: str,
          dataset_label: str = "") -> None:
    """Train the GCN; save weights and loss-curve PNG."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model     = build_model(in_channels=data.num_node_features).to(device)
    data      = data.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    criterion = torch.nn.CrossEntropyLoss()

    losses: list[float] = []
    model.train()
    for epoch in range(1, CONFIG["epochs"] + 1):
        optimizer.zero_grad()
        loss = criterion(
            model(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{CONFIG['epochs']} — Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), model_file)
    print(f"  Model saved to {model_file}")

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, CONFIG["epochs"] + 1), losses, linewidth=1.5)
    plt.xlabel("Epoch"); plt.ylabel("Cross-Entropy Loss")
    plt.title(f"Training Loss Curve — GCN ({dataset_label})")
    plt.tight_layout()
    plt.savefig(loss_curve_file, dpi=150)
    plt.close()
    print(f"  Loss curve saved to {loss_curve_file}")


def validate(data: Data, model_file: str, confusion_file: str,
             class_names: list[str], dataset_label: str = "") -> None:
    """Load checkpoint, evaluate on val_mask, save confusion matrix PNG."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(in_channels=data.num_node_features).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    print(f"  Loaded weights from {model_file}")

    data = data.to(device)
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index).argmax(dim=1)

    val_preds  = preds[data.val_mask].cpu().numpy()
    val_labels = data.y[data.val_mask].cpu().numpy()

    print(f"\n  Classification Report (Validation Set — {dataset_label}):")
    print(classification_report(val_labels, val_preds,
                                target_names=class_names, digits=4))

    cm  = confusion_matrix(val_labels, val_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im  = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=15)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — Validation ({dataset_label})")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(confusion_file, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {confusion_file}")


def preprocess_adult() -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess the Adult Census dataset."""
    df = pd.read_csv(
        ADULT_DATA_FILE, header=None, names=ADULT_COLUMNS,
        skipinitialspace=True, na_values="?",
    )
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Remap target: <=50K → 0, >50K → 1
    df["income"] = df["income"].str.strip().map({"<=50K": 0, ">50K": 1})
    y = df.pop("income").values.astype(np.int64)

    cat_cols = [c for c in df.columns if c not in ADULT_NUMERICAL_COLS]
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df[ADULT_NUMERICAL_COLS] = StandardScaler().fit_transform(df[ADULT_NUMERICAL_COLS])
    X = df.values.astype(np.float32)

    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"  Samples  : {X.shape[0]} | Features: {X.shape[1]}")
    print(f"  Classes  : 0 (<=50K)={dist.get(0,0)}, 1 (>50K)={dist.get(1,0)}")
    return X, y


# ===========================================================================
# Entry Point
# ===========================================================================

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    # ------------------------------------------------------------------
    print("=== Setup: Dataset Acquisition ===")
    print("\n-- German Credit Dataset --")
    download_german_data()

    print("\n-- Adult Census Income Dataset --")
    download_adult_data()

    # ------------------------------------------------------------------
    print("\n=== Phase 1: Training (German Credit) ===")
    X_g, y_g = preprocess_german()
    data_g    = build_graph(X_g, y_g)
    train(data_g,
          model_file=MODEL_FILE,
          loss_curve_file=LOSS_CURVE_FILE,
          dataset_label="German Credit")

    # ------------------------------------------------------------------
    print("\n=== Phase 2: Validation (German Credit) ===")
    validate(data_g,
             model_file=MODEL_FILE,
             confusion_file=CONFUSION_MATRIX_FILE,
             class_names=["Good Credit (0)", "Bad Credit (1)"],
             dataset_label="German Credit")


if __name__ == "__main__":
    main()
