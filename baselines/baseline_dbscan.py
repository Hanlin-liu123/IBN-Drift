
import os
import sys
import json
import time
import argparse
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLAUSE_NAMES = ['perf', 'path', 'energy']



def extract_flat_features(sample):

    window = sample.get('window', [])
    if not window:
        return None
    
    features = []

    link_features_all = []
    net_features_all = []
    
    for snapshot in window:
        link_vals = []
        links = snapshot.get('links', {})
        for link_id, link_data in sorted(links.items()):
            if isinstance(link_data, dict):
                link_vals.extend([
                    link_data.get('delay_ms', 0),
                    link_data.get('throughput_mbps', 0),
                    link_data.get('loss_rate', 0),
                    link_data.get('utilization', 0),
                    link_data.get('power_watts', 0),
                ])
        link_features_all.append(link_vals)

        net = snapshot.get('network', {})
        energy = snapshot.get('energy_metrics', {})
        net_vals = [
            net.get('total_throughput_mbps', 0),
            net.get('total_power_watts', energy.get('total_network_power', 0)),
            net.get('energy_efficiency', energy.get('energy_efficiency', 0)),
        ]
        net_features_all.append(net_vals)
    
    if not link_features_all or not link_features_all[0]:
        for snapshot in window:
            path_vals = []
            paths = snapshot.get('paths', {})
            for path_id, path_data in sorted(paths.items()):
                if isinstance(path_data, dict):
                    path_vals.extend([
                        path_data.get('e2e_delay_ms', 0),
                        path_data.get('e2e_throughput_mbps', 0),
                        path_data.get('e2e_loss_rate', 0),
                    ])
            link_features_all.append(path_vals)

    max_len = max(len(v) for v in link_features_all) if link_features_all else 0
    for i in range(len(link_features_all)):
        if len(link_features_all[i]) < max_len:
            link_features_all[i].extend([0.0] * (max_len - len(link_features_all[i])))
    
    link_arr = np.array(link_features_all, dtype=np.float32)  # [T, D_link]
    net_arr = np.array(net_features_all, dtype=np.float32)    # [T, D_net]

    last_link = link_arr[-1] if len(link_arr) > 0 else np.array([])
    last_net = net_arr[-1] if len(net_arr) > 0 else np.array([])

    link_mean = link_arr.mean(axis=0) if len(link_arr) > 0 else np.array([])
    link_std = link_arr.std(axis=0) if len(link_arr) > 0 else np.array([])

    feat = np.concatenate([
        v for v in [last_link, last_net, link_mean, link_std] if len(v) > 0
    ])

    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feat


def extract_labels(sample):

    fcl = sample.get('future_clause_labels', {})
    if not fcl:
        fcl = sample.get('current_clause_labels', {})
    
    clause_labels = {name: int(fcl.get(name, 0)) for name in CLAUSE_NAMES}
    any_drift = int(any(clause_labels[n] == 1 for n in CLAUSE_NAMES))
    
    return clause_labels, any_drift



class DBSCANDriftDetector:

    def __init__(self, eps=0.5, min_samples=5, multiplier=1.5):
        self.eps = eps
        self.min_samples = min_samples
        self.multiplier = multiplier
        self.scaler = StandardScaler()
        self.n_baseline_clusters = 0
        self.train_features = None
        self.train_labels = None
    
    def fit(self, train_data):

        print("  [DBSCAN] Extracting training features...")
        features = []
        labels = []
        normal_features = []
        
        for sample in train_data:
            feat = extract_flat_features(sample)
            if feat is None:
                continue
            clause_labels, any_drift = extract_labels(sample)
            features.append(feat)
            labels.append(any_drift)
            if any_drift == 0:
                normal_features.append(feat)
        
        if not features:
            print("  [DBSCAN] Error: no features extracted!")
            return

        max_dim = max(len(f) for f in features)
        features = [np.pad(f, (0, max_dim - len(f))) if len(f) < max_dim else f 
                     for f in features]
        normal_features = [np.pad(f, (0, max_dim - len(f))) if len(f) < max_dim else f 
                           for f in normal_features]
        
        self.train_features = np.array(features)
        self.train_labels = np.array(labels)

        self.scaler.fit(self.train_features)
        

        if normal_features:
            normal_scaled = self.scaler.transform(np.array(normal_features))
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            db.fit(normal_scaled)

            self.n_baseline_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        else:
            self.n_baseline_clusters = 1
        
        print(f"  [DBSCAN] Training samples: {len(features)} "
              f"(normal: {len(normal_features)}, drift: {sum(labels)})")
        print(f"  [DBSCAN] Feature dim: {max_dim}")
        print(f"  [DBSCAN] Baseline clusters (normal): {self.n_baseline_clusters}")

        all_scaled = self.scaler.transform(self.train_features)
        self.db_model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.db_model.fit(all_scaled)
        self.train_scaled = all_scaled
        
        n_all_clusters = len(set(self.db_model.labels_)) - (1 if -1 in self.db_model.labels_ else 0)
        n_noise = (self.db_model.labels_ == -1).sum()
        print(f"  [DBSCAN] All-data clusters: {n_all_clusters}, noise points: {n_noise}")
    
    def predict(self, test_data):

        print("  [DBSCAN] Predicting on test set...")
        
        all_preds = []
        all_clause_labels = {name: [] for name in CLAUSE_NAMES}
        all_any_labels = []
        
        for sample in test_data:
            feat = extract_flat_features(sample)
            clause_labels, any_drift = extract_labels(sample)
            
            for name in CLAUSE_NAMES:
                all_clause_labels[name].append(clause_labels[name])
            all_any_labels.append(any_drift)
            
            if feat is None:
                all_preds.append(0)
                continue
            

            expected_dim = self.train_scaled.shape[1]
            if len(feat) < expected_dim:
                feat = np.pad(feat, (0, expected_dim - len(feat)))
            elif len(feat) > expected_dim:
                feat = feat[:expected_dim]
            
            feat_scaled = self.scaler.transform(feat.reshape(1, -1))

            distances = np.linalg.norm(self.train_scaled - feat_scaled, axis=1)
            min_dist = distances.min()

            is_drift = 1 if min_dist > self.eps * self.multiplier else 0
            all_preds.append(is_drift)
        
        return all_preds, all_clause_labels, all_any_labels



def compute_detection_metrics(preds, labels):

    preds = np.asarray(preds)
    labels = np.asarray(labels)
    
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(len(preds), 1)
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'pos_samples': int(labels.sum()),
        'neg_samples': int(len(labels) - labels.sum()),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }


def auto_tune_eps(train_features, scaler, percentile=90):

    scaled = scaler.transform(train_features)

    n = min(500, len(scaled))
    idx = np.random.choice(len(scaled), n, replace=False)
    subset = scaled[idx]
    
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(subset)
    distances, _ = nn.kneighbors(subset)

    k_distances = distances[:, -1]
    eps = float(np.percentile(k_distances, percentile))
    
    return eps




def main():
    parser = argparse.ArgumentParser(description='DBSCAN drift detection baseline')
    parser.add_argument('--train', type=str, default=r'D:\datasets\Intent_drift\icsme\train_.json',
                        help='Path to training data JSON')
    parser.add_argument('--test', type=str, default=r'D:\datasets\Intent_drift\icsme\test_.json',
                        help='Path to test data JSON')
    parser.add_argument('--output', type=str, default='checkpoints1/results_dbscan.json',
                        help='Output path for results JSON')
    parser.add_argument('--eps', type=float, default=None,
                        help='DBSCAN eps parameter (default: auto-tune)')
    parser.add_argument('--min-samples', type=int, default=5,
                        help='DBSCAN min_samples parameter')
    parser.add_argument('--multiplier', type=float, default=1.0,
                        help='Distance multiplier for drift threshold')
    args = parser.parse_args()
    

    print(f"Loading training data: {args.train}")
    with open(args.train, 'r') as f:
        train_data = json.load(f)
    print(f"  Train samples: {len(train_data)}")
    
    print(f"Loading test data: {args.test}")
    with open(args.test, 'r') as f:
        test_data = json.load(f)
    print(f"  Test samples: {len(test_data)}")

    if args.eps is None:
        print("\n  Auto-tuning eps...")
        temp_features = []
        rng = np.random.default_rng(42)
        idx = rng.choice(len(train_data),size=min(2000, len(train_data)),replace=False)
        for i in idx:
            feat = extract_flat_features(train_data[i])

            if feat is not None:
                temp_features.append(feat)
        
        if temp_features:
            max_dim = max(len(f) for f in temp_features)
            temp_features = [np.pad(f, (0, max_dim - len(f))) if len(f) < max_dim else f 
                             for f in temp_features]
            temp_arr = np.array(temp_features)
            temp_scaler = StandardScaler()
            temp_scaler.fit(temp_arr)
            eps = auto_tune_eps(temp_arr, temp_scaler)
            print(f"  Auto-tuned eps: {eps:.4f}")
        else:
            eps = 1.0
            print(f"  Using default eps: {eps}")
    else:
        eps = args.eps
    

    print("\n" + "=" * 60)
    print("Training DBSCAN baseline")
    print("=" * 60)
    
    t0 = time.time()
    detector = DBSCANDriftDetector(
        eps=eps,
        min_samples=args.min_samples,
        multiplier=args.multiplier
    )
    detector.fit(train_data)
    t_train = time.time() - t0

    print("\n" + "=" * 60)
    print("Evaluating on test set")
    print("=" * 60)
    
    t0 = time.time()
    preds, clause_labels, any_labels = detector.predict(test_data)
    t_eval = time.time() - t0

    det_metrics = compute_detection_metrics(preds, any_labels)

    print(f"\n  DBSCAN Drift Detection [Muonagor et al.]")
    print(f"  {'='*50}")
    print(f"  eps={eps:.4f}, min_samples={args.min_samples}, multiplier={args.multiplier}")
    print(f"\n  Overall drift detection:")
    print(f"    Precision: {det_metrics['precision']:.4f}")
    print(f"    Recall:    {det_metrics['recall']:.4f}")
    print(f"    F1:        {det_metrics['f1']:.4f}")
    print(f"    Accuracy:  {det_metrics['accuracy']:.4f}")
    print(f"    Pos/Neg:   {det_metrics['pos_samples']}/{det_metrics['neg_samples']}")
    print(f"    TP={det_metrics['tp']} FP={det_metrics['fp']} "
          f"FN={det_metrics['fn']} TN={det_metrics['tn']}")
    print(f"\n  Macro F1:      N/A (unsupervised, no clause-level output)")
    print(f"  Per-clause F1: N/A")
    print(f"\n  Training time: {t_train:.1f}s")
    print(f"  Evaluation time: {t_eval:.1f}s")

    output = {
        'method': 'DBSCAN Drift Detection [Muonagor et al. LATINCOM 2024]',
        'test_data': args.test,
        'params': {
            'eps': eps,
            'min_samples': args.min_samples,
            'multiplier': args.multiplier,
        },
        'description': (
            'Unsupervised baseline: DBSCAN clustering on network features. '
            'Drift is detected when test samples fall outside known clusters. '
            'Corresponds to Algorithm 1 in Muonagor et al. (2024). '
            'No clause-level output capability.'
        ),
        'detection': det_metrics,
        'macro_f1': None,
        'per_clause': None,
    }
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {args.output}")


if __name__ == '__main__':
    main()
