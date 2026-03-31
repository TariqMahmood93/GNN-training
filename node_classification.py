"""
GNN Node Classification Pipeline — Registry-Driven
====================================================
Change ONE line to switch datasets:

    DATASET_NAME = "german_credit"   # ← or "adult_census", or any future entry

Everything else — schema, categorical values, target mapping, synthetic
distributions, class names, paths — is resolved automatically from the
REGISTRY below.  To add a new dataset, add one entry to REGISTRY.
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

# ===========================================================================
# ► THE ONLY LINE YOU NEED TO CHANGE
# ===========================================================================
DATASET_NAME = "german_credit"   # options: "german_credit" | "adult_census"
# ===========================================================================

# ---------------------------------------------------------------------------
# GNN hyper-parameters (shared across all datasets)
# ---------------------------------------------------------------------------
CONFIG = {
    "hidden_dim":   64,
    "dropout":      0.5,
    "lr":           0.01,
    "weight_decay": 5e-4,
    "epochs":       200,
    "k_neighbors":  5,
}

# ===========================================================================
# DATASET REGISTRY
# ===========================================================================
# Each entry is a self-contained description of one dataset.
#
# Required keys
# -------------
# dir          : output folder under datasets/
# url          : UCI (or other) download URL for the raw file
# sep          : field separator in the raw file (" " or ",")
# columns      : ordered list of all column names
# target_col   : name of the target column
# target_map   : dict mapping raw target values → 0 / 1
# class_names  : ["label for 0", "label for 1"]
# numerical_cols: columns to standardise
# n_synthetic  : rows to generate when download is unavailable
# columns_spec : per-column synthetic generation spec (see _SPEC_TYPES below)
#
# _SPEC_TYPES
# -----------
# {"type": "categorical",  "values": [...], "weights": [...]}  ← weighted choice
# {"type": "int_uniform",  "low": a, "high": b}               ← randint [a, b)
# {"type": "int_normal",   "mean": m, "std": s,
#                          "min": lo, "max": hi}               ← clipped normal int
# {"type": "sparse_int",   "low": a, "high": b,
#                          "zero_prob": p}                     ← mostly 0, else randint
# ===========================================================================

REGISTRY: dict[str, dict] = {

    # -----------------------------------------------------------------------
    "german_credit": {
        "dir": os.path.join("datasets", "german_credit"),
        "url": (
            "https://archive.ics.uci.edu/ml/machine-learning-databases"
            "/statlog/german/german.data"
        ),
        "sep": " ",
        "columns": [
            "checking_account", "duration",       "credit_history",
            "purpose",          "credit_amount",  "savings_account",
            "employment",       "installment_rate","personal_status",
            "other_debtors",    "residence_since", "property",
            "age",              "other_installments","housing",
            "existing_credits", "job",             "num_dependents",
            "telephone",        "foreign_worker",  "target",
        ],
        "target_col":     "target",
        "target_map":     {1: 0, 2: 1},        # 1=good→0, 2=bad→1
        "class_names":    ["Good Credit (0)", "Bad Credit (1)"],
        "numerical_cols": [
            "duration", "credit_amount", "installment_rate",
            "residence_since", "age", "existing_credits", "num_dependents",
        ],
        "n_synthetic": 1000,
        "columns_spec": {
            "checking_account":   {"type": "categorical",
                                   "values": ["A11","A12","A13","A14"]},
            "duration":           {"type": "int_uniform", "low": 4,   "high": 72},
            "credit_history":     {"type": "categorical",
                                   "values": ["A30","A31","A32","A33","A34"]},
            "purpose":            {"type": "categorical",
                                   "values": ["A40","A41","A42","A43","A44",
                                              "A45","A46","A47","A48","A49","A410"]},
            "credit_amount":      {"type": "int_uniform", "low": 250, "high": 18500},
            "savings_account":    {"type": "categorical",
                                   "values": ["A61","A62","A63","A64","A65"]},
            "employment":         {"type": "categorical",
                                   "values": ["A71","A72","A73","A74","A75"]},
            "installment_rate":   {"type": "int_uniform", "low": 1,   "high": 5},
            "personal_status":    {"type": "categorical",
                                   "values": ["A91","A92","A93","A94","A95"]},
            "other_debtors":      {"type": "categorical",
                                   "values": ["A101","A102","A103"]},
            "residence_since":    {"type": "int_uniform", "low": 1,   "high": 5},
            "property":           {"type": "categorical",
                                   "values": ["A121","A122","A123","A124"]},
            "age":                {"type": "int_uniform", "low": 19,  "high": 75},
            "other_installments": {"type": "categorical",
                                   "values": ["A141","A142","A143"]},
            "housing":            {"type": "categorical",
                                   "values": ["A151","A152","A153"]},
            "existing_credits":   {"type": "int_uniform", "low": 1,   "high": 5},
            "job":                {"type": "categorical",
                                   "values": ["A171","A172","A173","A174"]},
            "num_dependents":     {"type": "int_uniform", "low": 1,   "high": 3},
            "telephone":          {"type": "categorical",
                                   "values": ["A191","A192"]},
            "foreign_worker":     {"type": "categorical",
                                   "values": ["A201","A202"]},
            "target":             {"type": "categorical",
                                   "values": [1, 2],
                                   "weights": [0.70, 0.30]},
        },
    },

    # -----------------------------------------------------------------------
    "adult_census": {
        "dir": os.path.join("datasets", "adult_census"),
        "url": (
            "https://archive.ics.uci.edu/ml/machine-learning-databases"
            "/adult/adult.data"
        ),
        "sep": ",",
        "columns": [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week",
            "native_country", "income",
        ],
        "target_col":  "income",
        "target_map":  {"<=50K": 0, ">50K": 1},
        "class_names": ["<=50K (0)", ">50K (1)"],
        "numerical_cols": [
            "age", "fnlwgt", "education_num",
            "capital_gain", "capital_loss", "hours_per_week",
        ],
        "n_synthetic": 48_842,
        "columns_spec": {
            "age":            {"type": "int_normal",  "mean": 38.6, "std": 13.6,
                               "min": 17, "max": 90},
            "workclass":      {"type": "categorical",
                               "values":  ["Private","Self-emp-not-inc","Self-emp-inc",
                                           "Federal-gov","Local-gov","State-gov",
                                           "Without-pay","Never-worked"],
                               "weights": [0.690, 0.080, 0.030, 0.030, 0.060,
                                           0.040, 0.005, 0.005]},
            "fnlwgt":         {"type": "int_uniform", "low": 13_000, "high": 1_500_000},
            "education":      {"type": "categorical",
                               "values":  ["Bachelors","Some-college","11th","HS-grad",
                                           "Prof-school","Assoc-acdm","Assoc-voc","9th",
                                           "7th-8th","12th","Masters","1st-4th","10th",
                                           "Doctorate","5th-6th","Preschool"],
                               "weights": [0.164, 0.223, 0.039, 0.316, 0.018, 0.033,
                                           0.042, 0.016, 0.020, 0.014, 0.053, 0.006,
                                           0.015, 0.013, 0.009, 0.002]},
            "education_num":  {"type": "int_normal",  "mean": 10.1, "std": 2.6,
                               "min": 1, "max": 16},
            "marital_status": {"type": "categorical",
                               "values":  ["Married-civ-spouse","Divorced","Never-married",
                                           "Separated","Widowed",
                                           "Married-spouse-absent","Married-AF-spouse"],
                               "weights": [0.459, 0.135, 0.328, 0.032, 0.033,
                                           0.010, 0.003]},
            "occupation":     {"type": "categorical",
                               "values":  ["Tech-support","Craft-repair","Other-service",
                                           "Sales","Exec-managerial","Prof-specialty",
                                           "Handlers-cleaners","Machine-op-inspct",
                                           "Adm-clerical","Farming-fishing",
                                           "Transport-moving","Priv-house-serv",
                                           "Protective-serv","Armed-Forces"],
                               "weights": [0.029, 0.123, 0.107, 0.076, 0.099, 0.126,
                                           0.042, 0.062, 0.114, 0.031, 0.049, 0.015,
                                           0.020, 0.003]},
            "relationship":   {"type": "categorical",
                               "values":  ["Wife","Own-child","Husband",
                                           "Not-in-family","Other-relative","Unmarried"],
                               "weights": [0.105, 0.188, 0.401, 0.155, 0.032, 0.119]},
            "race":           {"type": "categorical",
                               "values":  ["White","Asian-Pac-Islander",
                                           "Amer-Indian-Eskimo","Other","Black"],
                               "weights": [0.855, 0.031, 0.010, 0.008, 0.096]},
            "sex":            {"type": "categorical",
                               "values":  ["Male","Female"],
                               "weights": [0.669, 0.331]},
            "capital_gain":   {"type": "sparse_int",  "low": 1, "high": 99_999,
                               "zero_prob": 0.917},
            "capital_loss":   {"type": "sparse_int",  "low": 1, "high": 4_356,
                               "zero_prob": 0.953},
            "hours_per_week": {"type": "int_normal",  "mean": 40.4, "std": 12.3,
                               "min": 1, "max": 99},
            "native_country": {"type": "categorical",
                               "values":  [
                                   "United-States","Cuba","Jamaica","India","Mexico",
                                   "South","Japan","Philippines","Germany","Canada",
                                   "Puerto-Rico","El-Salvador","Greece","Italy","China",
                                   "South-Korea","Iran","Poland","Columbia","Haiti",
                                   "Portugal","Taiwan","Hungary","Honduras","Ecuador",
                                   "Peru","France","Guatemala","Nicaragua","Vietnam",
                                   "Trinadad&Tobago","Laos","Thailand","Yugoslavia",
                                   "Dominican-Republic","Cambodia","Ireland","Hong",
                                   "Scotland","Outlying-US(Guam-USVI-etc)",
                                   "England","Holland-Netherlands"],
                               "weights": (
                                   lambda vals: [0.90] + [0.10 / 41] * 41
                               )(None)},
            "income":         {"type": "categorical",
                               "values":  ["<=50K", ">50K"],
                               "weights": [0.7592, 0.2408]},
        },
    },
}

# ===========================================================================
# Generic synthetic data generator  (reads the registry spec — no hardcoding)
# ===========================================================================

def _make_column(spec: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n values for one column according to its spec."""
    t = spec["type"]

    if t == "categorical":
        vals = spec["values"]
        w    = spec.get("weights")
        if w is not None:
            w = np.array(w, dtype=float)
            w /= w.sum()
        return rng.choice(vals, size=n, p=w)

    if t == "int_uniform":
        return rng.integers(spec["low"], spec["high"] + 1, size=n)

    if t == "int_normal":
        arr = rng.normal(spec["mean"], spec["std"], n).round().astype(int)
        return np.clip(arr, spec["min"], spec["max"])

    if t == "sparse_int":
        mask = rng.random(n) < spec["zero_prob"]
        vals = rng.integers(spec["low"], spec["high"] + 1, size=n)
        return np.where(mask, 0, vals)

    raise ValueError(f"Unknown column spec type: {t!r}")


def _generate_synthetic(cfg: dict, path: str) -> None:
    """
    Generate a synthetic dataset from the registry spec and write it to `path`.
    Only the `columns_spec` dict drives generation — no dataset-specific code here.
    """
    rng = np.random.default_rng(42)
    n   = cfg["n_synthetic"]

    data = {
        col: _make_column(cfg["columns_spec"][col], n, rng)
        for col in cfg["columns"]
    }

    sep = cfg["sep"]
    pd.DataFrame(data, columns=cfg["columns"]).to_csv(
        path, sep=sep, index=False, header=False
    )
    print(f"  Synthetic data written → {path}  ({n:,} rows)")


# ===========================================================================
# Step 1 — Dataset Acquisition
# ===========================================================================

def download_data(cfg: dict) -> None:
    """
    Try to download the raw dataset from UCI.
    Falls back to the registry-driven synthetic generator if unavailable.
    Always (re)writes a tidy CSV with column headers.
    """
    os.makedirs(cfg["dir"], exist_ok=True)
    raw_file = os.path.join(cfg["dir"], os.path.basename(cfg["url"]))
    csv_file = raw_file.replace(
        os.path.splitext(raw_file)[1], ".csv"
    )
    # also expose these on the cfg so later steps can find the files
    cfg["_raw_file"] = raw_file
    cfg["_csv_file"] = csv_file

    if os.path.exists(raw_file):
        print(f"  Already exists: {raw_file}")
    else:
        print(f"  Downloading from {cfg['url']} …")
        try:
            urllib.request.urlretrieve(cfg["url"], raw_file)
            print(f"  Saved → {raw_file}")
        except Exception as exc:
            print(f"  Download failed ({exc}). Using synthetic fallback …")
            _generate_synthetic(cfg, raw_file)

    # Load raw file and write tidy CSV
    df = pd.read_csv(
        raw_file,
        sep=cfg["sep"],
        header=None,
        names=cfg["columns"],
        skipinitialspace=True,
        na_values="?",
    )
    df.dropna(inplace=True)
    df.to_csv(csv_file, index=False)
    print(f"  CSV saved → {csv_file}")

    # Print dataset statistics
    tgt  = cfg["target_col"]
    dist = df[tgt].value_counts().to_dict()
    n_features = len(cfg["columns"]) - 1
    print(f"  Samples  : {len(df):,} | Features: {n_features}")
    print(f"  Target   : {tgt} distribution → {dist}")


# ===========================================================================
# Step 2 — Preprocessing
# ===========================================================================

def preprocess(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the raw file, encode categoricals, scale numericals,
    apply target mapping.  Fully driven by cfg — no dataset-specific code.
    """
    df = pd.read_csv(
        cfg["_raw_file"],
        sep=cfg["sep"],
        header=None,
        names=cfg["columns"],
        skipinitialspace=True,
        na_values="?",
    )
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Target
    tgt = cfg["target_col"]
    df[tgt] = df[tgt].map(cfg["target_map"])
    # Handle targets that are already 0/1 ints but survive .map as floats
    df.dropna(subset=[tgt], inplace=True)
    y = df.pop(tgt).values.astype(np.int64)

    # Categorical encoding
    cat_cols = [c for c in df.columns if c not in cfg["numerical_cols"]]
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Numerical scaling
    df[cfg["numerical_cols"]] = StandardScaler().fit_transform(
        df[cfg["numerical_cols"]]
    )

    X = df.values.astype(np.float32)
    unique, counts = np.unique(y, return_counts=True)
    print(f"  Encoded  : {X.shape[0]:,} samples × {X.shape[1]} features")
    print(f"  Labels   : {dict(zip(cfg['class_names'], counts))}")
    return X, y


# ===========================================================================
# Step 3 — Graph Construction + Node Masks
# ===========================================================================

def build_graph(X: np.ndarray, y: np.ndarray) -> Data:
    """Build a KNN graph (k from CONFIG) and attach 60/20/20 node masks."""
    k = CONFIG["k_neighbors"]
    print(f"  Building KNN graph  k={k} …")

    A  = kneighbors_graph(X, n_neighbors=k, mode="connectivity", include_self=False)
    cx = A.tocoo()
    edge_index = torch.tensor(np.vstack([cx.row, cx.col]), dtype=torch.long)

    data = Data(
        x          = torch.tensor(X, dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor(y, dtype=torch.long),
    )

    n     = data.num_nodes
    perm  = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    n_tr  = int(0.60 * n)
    n_val = int(0.20 * n)

    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.val_mask   = torch.zeros(n, dtype=torch.bool)
    data.test_mask  = torch.zeros(n, dtype=torch.bool)
    data.train_mask[perm[:n_tr]]             = True
    data.val_mask  [perm[n_tr:n_tr + n_val]] = True
    data.test_mask [perm[n_tr + n_val:]]     = True

    print(f"  Nodes: {n:,} | Edges: {edge_index.size(1):,}")
    print(f"  Split → Train: {data.train_mask.sum().item()} | "
          f"Val: {data.val_mask.sum().item()} | "
          f"Test: {data.test_mask.sum().item()}")
    return data


# ===========================================================================
# Step 4 — Model
# ===========================================================================

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
        in_channels  = in_channels,
        hidden_dim   = CONFIG["hidden_dim"],
        out_channels = num_classes,
        dropout      = CONFIG["dropout"],
    )


# ===========================================================================
# Phase 1 — Training
# ===========================================================================

def train(data: Data, cfg: dict) -> None:
    """Train the GCN; save model checkpoint and loss-curve PNG."""
    model_file      = os.path.join(cfg["dir"], "trained_model.pt")
    loss_curve_file = os.path.join(cfg["dir"], "training_loss_curve.png")
    cfg["_model_file"] = model_file          # pass to validate()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    print(f"  Model saved → {model_file}")

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, CONFIG["epochs"] + 1), losses, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"Training Loss — GCN ({DATASET_NAME})")
    plt.tight_layout()
    plt.savefig(loss_curve_file, dpi=150)
    plt.close()
    print(f"  Loss curve → {loss_curve_file}")


# ===========================================================================
# Phase 2 — Validation
# ===========================================================================

def validate(data: Data, cfg: dict) -> None:
    """Load checkpoint, evaluate on val_mask, print report, save confusion matrix."""
    model_file      = cfg["_model_file"]
    confusion_file  = os.path.join(cfg["dir"], "validation_confusion_matrix.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(in_channels=data.num_node_features).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    print(f"  Loaded weights from {model_file}")

    data = data.to(device)
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index).argmax(dim=1)

    val_preds  = preds[data.val_mask].cpu().numpy()
    val_labels = data.y[data.val_mask].cpu().numpy()

    print(f"\n  Classification Report — {DATASET_NAME} (Validation):")
    print(classification_report(val_labels, val_preds,
                                target_names=cfg["class_names"], digits=4))

    # Confusion matrix
    cm     = confusion_matrix(val_labels, val_preds)
    names  = cfg["class_names"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {DATASET_NAME} (Validation)")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(confusion_file, dpi=150)
    plt.close()
    print(f"  Confusion matrix → {confusion_file}")


# ===========================================================================
# Entry Point
# ===========================================================================

def main() -> None:
    if DATASET_NAME not in REGISTRY:
        raise ValueError(
            f"Unknown dataset {DATASET_NAME!r}. "
            f"Available: {list(REGISTRY.keys())}"
        )

    torch.manual_seed(42)
    np.random.seed(42)

    cfg = REGISTRY[DATASET_NAME]

    print("=== Setup: Dataset Acquisition ===")
    download_data(cfg)

    print("\n=== Phase 1: Training ===")
    X, y = preprocess(cfg)
    data = build_graph(X, y)
    train(data, cfg)

    print("\n=== Phase 2: Validation ===")
    validate(data, cfg)


if __name__ == "__main__":
    main()
