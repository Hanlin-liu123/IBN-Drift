

import os
import sys
import json
import time
import argparse
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.impute import SimpleImputer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLAUSE_NAMES = ["perf", "path", "energy"]


# ============================================================
# Feature extraction
# ============================================================

def extract_flat_features(sample):
    """
    Extract a flat feature vector from one sequence sample.

    The design is kept close to baseline_dbscan.py:
      - use link-level features over the observation window;
      - concatenate the last-step features and temporal statistics;
      - append network-level features.
    """

    window = sample.get("window", [])
    if not window:
        return None

    link_features_all = []
    net_features_all = []

    for snapshot in window:
        link_vals = []
        links = snapshot.get("links", {})

        for link_id, link_data in sorted(links.items()):
            if isinstance(link_data, dict):
                link_vals.extend([
                    link_data.get("delay_ms", 0),
                    link_data.get("throughput_mbps", 0),
                    link_data.get("loss_rate", 0),
                    link_data.get("utilization", 0),
                    link_data.get("power_watts", 0),
                ])

        link_features_all.append(link_vals)

        net = snapshot.get("network", {})
        energy = snapshot.get("energy_metrics", {})

        net_vals = [
            net.get("total_throughput_mbps", 0),
            net.get("total_power_watts", energy.get("total_network_power", 0)),
            net.get("energy_efficiency", energy.get("energy_efficiency", 0)),
        ]

        net_features_all.append(net_vals)

    # Fallback: use path-level metrics if link-level metrics are missing.
    if not link_features_all or not link_features_all[0]:
        link_features_all = []

        for snapshot in window:
            path_vals = []
            paths = snapshot.get("paths", {})

            for path_id, path_data in sorted(paths.items()):
                if isinstance(path_data, dict):
                    path_vals.extend([
                        path_data.get("e2e_delay_ms", 0),
                        path_data.get("e2e_throughput_mbps", 0),
                        path_data.get("e2e_loss_rate", 0),
                    ])

            link_features_all.append(path_vals)

    if not link_features_all:
        return None

    # Ensure consistent feature dimension across timesteps.
    max_len = max(len(v) for v in link_features_all)

    for i in range(len(link_features_all)):
        if len(link_features_all[i]) < max_len:
            link_features_all[i].extend([0.0] * (max_len - len(link_features_all[i])))

    link_arr = np.array(link_features_all, dtype=np.float32)
    net_arr = np.array(net_features_all, dtype=np.float32)

    last_link = link_arr[-1] if len(link_arr) > 0 else np.array([])
    last_net = net_arr[-1] if len(net_arr) > 0 else np.array([])

    link_mean = link_arr.mean(axis=0) if len(link_arr) > 0 else np.array([])
    link_std = link_arr.std(axis=0) if len(link_arr) > 0 else np.array([])

    net_mean = net_arr.mean(axis=0) if len(net_arr) > 0 else np.array([])
    net_std = net_arr.std(axis=0) if len(net_arr) > 0 else np.array([])

    feat = np.concatenate([
        v for v in [
            last_link,
            last_net,
            link_mean,
            link_std,
            net_mean,
            net_std,
        ]
        if len(v) > 0
    ])

    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    return feat


def extract_labels(sample):
    """
    Extract future constraint-type labels and overall drift label.
    """

    fcl = sample.get("future_clause_labels", {})

    if not fcl:
        fcl = sample.get("current_clause_labels", {})

    clause_labels = {
        name: int(fcl.get(name, 0))
        for name in CLAUSE_NAMES
    }

    any_drift = int(any(clause_labels[name] == 1 for name in CLAUSE_NAMES))

    return clause_labels, any_drift


def build_feature_matrix(data):
    """
    Convert a list of sequence samples into X, y_any, y_clause.
    """

    features = []
    y_any = []
    y_clause = {name: [] for name in CLAUSE_NAMES}
    kept = 0
    skipped = 0

    for sample in data:
        feat = extract_flat_features(sample)

        if feat is None:
            skipped += 1
            continue

        clause_labels, any_drift = extract_labels(sample)

        features.append(feat)
        y_any.append(any_drift)

        for name in CLAUSE_NAMES:
            y_clause[name].append(clause_labels[name])

        kept += 1

    if not features:
        raise RuntimeError("No valid features extracted.")

    max_dim = max(len(f) for f in features)

    padded_features = []
    for f in features:
        if len(f) < max_dim:
            f = np.pad(f, (0, max_dim - len(f)))
        elif len(f) > max_dim:
            f = f[:max_dim]
        padded_features.append(f)

    X = np.asarray(padded_features, dtype=np.float32)
    y_any = np.asarray(y_any, dtype=np.int64)
    y_clause = {
        name: np.asarray(y_clause[name], dtype=np.int64)
        for name in CLAUSE_NAMES
    }

    return X, y_any, y_clause, kept, skipped


def align_feature_dim(X, target_dim):
    """
    Align feature dimension between train and test.
    """

    if X.shape[1] < target_dim:
        pad_width = target_dim - X.shape[1]
        X = np.pad(X, ((0, 0), (0, pad_width)))
    elif X.shape[1] > target_dim:
        X = X[:, :target_dim]

    return X


# ============================================================
# Metrics
# ============================================================

def compute_binary_metrics(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    acc = accuracy_score(labels, preds)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "pos_samples": int(labels.sum()),
        "neg_samples": int(len(labels) - labels.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def print_metrics(title, metrics):
    print(f"\n  {title}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1:        {metrics['f1']:.4f}")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Pos/Neg:   {metrics['pos_samples']}/{metrics['neg_samples']}")
    print(
        f"    TP={metrics['tp']} FP={metrics['fp']} "
        f"FN={metrics['fn']} TN={metrics['tn']}"
    )


# ============================================================
# Random Forest baseline
# ============================================================

class RandomForestDriftBaseline:
    def __init__(
        self,
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.imputer = SimpleImputer(strategy="median")

        self.overall_model = None
        self.clause_models = {}

    def _make_model(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def fit(self, X_train, y_any_train, y_clause_train):
        print("  [RandomForest] Fitting imputer...")
        X_train = self.imputer.fit_transform(X_train)

        print("  [RandomForest] Training overall drift detector...")
        self.overall_model = self._make_model()
        self.overall_model.fit(X_train, y_any_train)

        print("  [RandomForest] Training constraint-type classifiers...")
        for name in CLAUSE_NAMES:
            y = y_clause_train[name]

            if len(np.unique(y)) < 2:
                print(f"    Warning: {name} has only one class in training data. "
                      f"Using constant predictor.")
                self.clause_models[name] = None
            else:
                model = self._make_model()
                model.fit(X_train, y)
                self.clause_models[name] = model

    def predict(self, X_test):
        X_test = self.imputer.transform(X_test)

        any_preds = self.overall_model.predict(X_test)

        clause_preds = {}

        for name in CLAUSE_NAMES:
            model = self.clause_models.get(name)

            if model is None:
                clause_preds[name] = np.zeros(X_test.shape[0], dtype=np.int64)
            else:
                clause_preds[name] = model.predict(X_test)

        return any_preds, clause_preds


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Random Forest supervised baseline for IBN-Drift."
    )

    parser.add_argument(
        "--train",
        type=str,
        default=r'D:\datasets\Intent_drift\icsme\train_T10_h3_1.json',
        help="Path to training data JSON.",
    )

    parser.add_argument(
        "--test",
        type=str,
        default=r'D:\datasets\Intent_drift\icsme\test_T10_h3_1.json',
        help="Path to test data JSON.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/results_random_forest.json",
        help="Output path for results JSON.",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees in the Random Forest.",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum tree depth. Default: None.",
    )

    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=2,
        help="Minimum samples per leaf.",
    )

    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        choices=["balanced", "balanced_subsample", "none"],
        help="Class weight strategy.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    class_weight = None if args.class_weight == "none" else args.class_weight

    print(f"Loading training data: {args.train}")
    with open(args.train, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    print(f"  Train samples: {len(train_data)}")

    print(f"Loading test data: {args.test}")
    with open(args.test, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"  Test samples: {len(test_data)}")

    print("\nExtracting features...")
    X_train, y_any_train, y_clause_train, kept_train, skipped_train = build_feature_matrix(train_data)
    X_test, y_any_test, y_clause_test, kept_test, skipped_test = build_feature_matrix(test_data)

    train_dim = X_train.shape[1]
    X_test = align_feature_dim(X_test, train_dim)

    print(f"  Train features: {X_train.shape}, kept={kept_train}, skipped={skipped_train}")
    print(f"  Test features:  {X_test.shape}, kept={kept_test}, skipped={skipped_test}")

    print("\nTraining label distribution:")
    print(f"  Any drift: {int(y_any_train.sum())}/{len(y_any_train)} "
          f"({y_any_train.mean():.1%})")

    for name in CLAUSE_NAMES:
        y = y_clause_train[name]
        print(f"  {name}: {int(y.sum())}/{len(y)} ({y.mean():.1%})")

    print("\n" + "=" * 60)
    print("Training Random Forest baseline")
    print("=" * 60)

    t0 = time.time()

    model = RandomForestDriftBaseline(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight=class_weight,
        random_state=args.random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_any_train, y_clause_train)

    t_train = time.time() - t0

    print("\n" + "=" * 60)
    print("Evaluating on test set")
    print("=" * 60)

    t0 = time.time()
    any_preds, clause_preds = model.predict(X_test)
    t_eval = time.time() - t0

    overall_metrics = compute_binary_metrics(any_preds, y_any_test)

    per_clause = {}
    for name in CLAUSE_NAMES:
        per_clause[name] = compute_binary_metrics(
            clause_preds[name],
            y_clause_test[name],
        )

    print("\n  Random Forest Drift Detection")
    print(f"  {'=' * 50}")
    print(f"  n_estimators={args.n_estimators}, "
          f"max_depth={args.max_depth}, "
          f"min_samples_leaf={args.min_samples_leaf}, "
          f"class_weight={class_weight}")

    print_metrics("Overall drift detection:", overall_metrics)

    print("\n  Per-dimension results:")
    print("  Dimension          Prec     Recall   F1       Acc      Pos    Neg")
    print("  ------------------------------------------------------------------")

    for name in CLAUSE_NAMES:
        m = per_clause[name]
        display_name = {
            "perf": "Performance",
            "path": "Path semantics",
            "energy": "Energy",
        }.get(name, name)

        print(
            f"  {display_name:<17} "
            f"{m['precision']:<8.3f} "
            f"{m['recall']:<8.3f} "
            f"{m['f1']:<8.3f} "
            f"{m['accuracy']:<8.3f} "
            f"{m['pos_samples']:<6d} "
            f"{m['neg_samples']:<6d}"
        )

    print(f"\n  Training time:   {t_train:.2f}s")
    print(f"  Evaluation time: {t_eval:.2f}s")

    output = {
        "method": "Random Forest supervised baseline",
        "train_data": args.train,
        "test_data": args.test,
        "params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "class_weight": class_weight,
            "random_state": args.random_state,
        },
        "description": (
            "Supervised baseline using flattened window-level network features. "
            "The model predicts overall intent drift and three constraint-type labels "
            "for performance, path semantics, and energy."
        ),
        "detection": overall_metrics,
        "per_clause": per_clause,
        "training_time_sec": float(t_train),
        "evaluation_time_sec": float(t_eval),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()