
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLAUSE_NAMES = ['perf', 'path', 'energy']


# ============================================================
# 特征提取
# ============================================================

def extract_features_and_labels(sample):

    window = sample.get('window', [])
    if len(window) < 2:
        return None, None, None
    
    last = window[-1]
    prev = window[-2]
    
    def extract_snapshot_kpis(snapshot):
        vals = []

        paths = snapshot.get('paths', {})
        e2e_delay = 0
        e2e_loss = 0
        e2e_throughput = 0
        e2e_jitter = 0
        
        for pid, pdata in paths.items():
            if isinstance(pdata, dict):
                e2e_delay = max(e2e_delay, pdata.get('e2e_delay_ms', 0) or 0)
                e2e_loss = max(e2e_loss, pdata.get('e2e_loss_rate', 0) or 0)
                e2e_throughput = max(e2e_throughput, pdata.get('e2e_throughput_mbps', 0) or 0)
                e2e_jitter = max(e2e_jitter, pdata.get('e2e_jitter_ms', 0) or 0)
                break
        
        vals.extend([e2e_delay, e2e_loss, e2e_throughput, e2e_jitter])

        net = snapshot.get('network', {})
        energy = snapshot.get('energy_metrics', {})
        
        total_throughput = net.get('total_throughput_mbps', 0) or 0
        total_power = (net.get('total_power_watts', 0) or
                       energy.get('total_network_power', 0) or 0)
        energy_efficiency = (net.get('energy_efficiency', 0) or
                             energy.get('energy_efficiency', 0) or 0)
        
        vals.extend([total_throughput, total_power, energy_efficiency])

        links = snapshot.get('links', {})
        link_delays = []
        link_losses = []
        link_utils = []
        link_powers = []
        
        for lid, ldata in links.items():
            if isinstance(ldata, dict):
                link_delays.append(ldata.get('delay_ms', 0) or 0)
                link_losses.append(ldata.get('loss_rate', 0) or 0)
                link_utils.append(ldata.get('utilization', 0) or 0)
                link_powers.append(ldata.get('power_watts', 0) or 0)
        
        if link_delays:
            vals.extend([
                np.mean(link_delays), np.max(link_delays), np.std(link_delays),
                np.mean(link_losses), np.max(link_losses),
                np.mean(link_utils), np.max(link_utils),
                np.mean(link_powers), np.sum(link_powers),
            ])
        else:
            vals.extend([0] * 9)
        
        return np.array(vals, dtype=np.float32)

    kpi_current = extract_snapshot_kpis(last)
    kpi_prev = extract_snapshot_kpis(prev)

    delta = kpi_current - kpi_prev

    features = np.concatenate([kpi_current, delta])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    fcl = sample.get('future_clause_labels', {})
    if not fcl:
        fcl = sample.get('current_clause_labels', {})
    
    clause_labels = {name: int(fcl.get(name, 0)) for name in CLAUSE_NAMES}
    any_drift = int(any(clause_labels[n] == 1 for n in CLAUSE_NAMES))
    
    return features, clause_labels, any_drift


def prepare_dataset(data):
    features_list = []
    clause_labels_list = {name: [] for name in CLAUSE_NAMES}
    any_labels_list = []
    
    for sample in data:
        feat, clause_labels, any_drift = extract_features_and_labels(sample)
        if feat is None:
            continue
        features_list.append(feat)
        for name in CLAUSE_NAMES:
            clause_labels_list[name].append(clause_labels[name])
        any_labels_list.append(any_drift)
    
    X = np.array(features_list, dtype=np.float32)
    y = np.array(any_labels_list, dtype=np.float32)
    
    return X, y, clause_labels_list


class RiskScoreMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def ema_smooth(scores, window=5):
    alpha = 2.0 / (window + 1)
    smoothed = np.zeros_like(scores)
    smoothed[0] = scores[0]
    for i in range(1, len(scores)):
        smoothed[i] = alpha * scores[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed



def tune_threshold_f1(scores, labels):
    best_f1 = 0
    best_tau = 0.5
    
    for tau in np.arange(0.05, 0.95, 0.01):
        preds = (scores >= tau).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
    
    return best_tau, best_f1



def train_lead_drift(X_train, y_train, X_val, y_val,
                     hidden_dim=64, epochs=30, batch_size=256,
                     lr=1e-3, ema_window=5, device='cpu'):
    device = torch.device(device)
    

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    

    train_ds = TensorDataset(
        torch.tensor(X_train_norm, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    

    model = RiskScoreMLP(X_train.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.MSELoss()
    
    best_val_f1 = 0
    best_state = None
    best_tau = 0.5
    patience = 10
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            n += len(xb)
        train_loss = total_loss / max(n, 1)

        model.eval()
        with torch.no_grad():
            val_scores = model(
                torch.tensor(X_val_norm, dtype=torch.float32).to(device)
            ).cpu().numpy()

        val_smoothed = ema_smooth(val_scores, window=ema_window)

        tau, val_f1 = tune_threshold_f1(val_smoothed, y_val)
        
        print(f"    [LEAD-Drift] Epoch {epoch+1}/{epochs}: "
              f"train_loss={train_loss:.4f} val_f1={val_f1:.4f} tau={tau:.3f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_tau = tau
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    [LEAD-Drift] Early stopping at epoch {epoch+1}")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, mean, std, best_tau


def evaluate_lead_drift(model, X_test, y_test, clause_labels_test,
                        mean, std, tau, ema_window=5, device='cpu'):
    device_t = torch.device(device)
    
    X_test_norm = (X_test - mean) / (std + 1e-8)
    
    model.eval()
    with torch.no_grad():
        raw_scores = model(
            torch.tensor(X_test_norm, dtype=torch.float32).to(device_t)
        ).cpu().numpy()
    

    smoothed = ema_smooth(raw_scores, window=ema_window)
    

    preds = (smoothed >= tau).astype(int)
    

    labels = y_test.astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    
    det_precision = tp / max(tp + fp, 1)
    det_recall = tp / max(tp + fn, 1)
    det_f1 = 2 * det_precision * det_recall / max(det_precision + det_recall, 1e-8)
    det_accuracy = (tp + tn) / max(len(labels), 1)
    
    detection = {
        'precision': float(det_precision),
        'recall': float(det_recall),
        'f1': float(det_f1),
        'accuracy': float(det_accuracy),
        'pos_samples': int(labels.sum()),
        'neg_samples': int(len(labels) - labels.sum()),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }
    
    return detection




def main():
    parser = argparse.ArgumentParser(description='LEAD-Drift baseline')
    parser.add_argument('--train', type=str, default=r'D:\datasets\Intent_drift\icsme\train.json')
    parser.add_argument('--val', type=str, default=r'D:\datasets\Intent_drift\icsme\val.json',
                        help='Validation data (default: split 15%% from train)')
    parser.add_argument('--test', type=str, default=r'D:\datasets\Intent_drift\icsme\test.json')
    parser.add_argument('--output', type=str, default='checkpoints/results_lead_drift.json')
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--ema-window', type=int, default=5,
                        help='EMA smoothing window ')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLoading training data: {args.train}")
    with open(args.train, 'r') as f:
        train_data = json.load(f)
    
    print(f"Loading test data: {args.test}")
    with open(args.test, 'r') as f:
        test_data = json.load(f)

    print("\nExtracting features...")
    X_train_full, y_train_full, _ = prepare_dataset(train_data)
    X_test, y_test, clause_labels_test = prepare_dataset(test_data)
    
    print(f"  Train: {X_train_full.shape[0]} samples, {X_train_full.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]} samples")
    print(f"  Train drift ratio: {y_train_full.mean():.2%}")
    print(f"  Test drift ratio:  {y_test.mean():.2%}")

    if args.val:
        with open(args.val, 'r') as f:
            val_data = json.load(f)
        X_val, y_val, _ = prepare_dataset(val_data)
    else:
        n = len(X_train_full)
        split = int(n * 0.85)
        X_train, y_train = X_train_full[:split], y_train_full[:split]
        X_val, y_val = X_train_full[split:], y_train_full[split:]
        X_train_full = X_train
        y_train_full = y_train
    
    print(f"  Val:   {X_val.shape[0]} samples")

    print("\n" + "=" * 60)
    print("Training LEAD-Drift MLP")
    print("=" * 60)
    
    t0 = time.time()
    model, mean, std, best_tau = train_lead_drift(
        X_train_full, y_train_full, X_val, y_val,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ema_window=args.ema_window,
        device=device,
    )
    t_train = time.time() - t0
    print(f"\n  Training time: {t_train:.1f}s")
    print(f"  Best threshold (tau): {best_tau:.3f}")

    print("\n" + "=" * 60)
    print("Evaluating on test set")
    print("=" * 60)
    
    t0 = time.time()
    detection = evaluate_lead_drift(
        model, X_test, y_test, clause_labels_test,
        mean, std, best_tau,
        ema_window=args.ema_window,
        device=device,
    )
    t_eval = time.time() - t0
    

    print(f"\n  LEAD-Drift [Hossain & Aljoby, ICC 2026]")
    print(f"  {'='*50}")
    print(f"\n  Overall drift detection:")
    print(f"    Precision: {detection['precision']:.4f}")
    print(f"    Recall:    {detection['recall']:.4f}")
    print(f"    F1:        {detection['f1']:.4f}")
    print(f"    Accuracy:  {detection['accuracy']:.4f}")
    print(f"    Pos/Neg:   {detection['pos_samples']}/{detection['neg_samples']}")
    print(f"\n  Macro F1:      N/A (single risk score, no clause-level output)")
    print(f"  Per-clause F1: N/A")
    print(f"\n  Evaluation time: {t_eval:.1f}s")

    output = {
        'method': 'LEAD-Drift [Hossain & Aljoby, IEEE ICC 2026]',
        'test_data': args.test,
        'params': {
            'hidden_dim': args.hidden_dim,
            'ema_window': args.ema_window,
            'threshold': float(best_tau),
        },
        'description': (
            'Proactive baseline: lightweight MLP predicts a scalar risk score from '
            'current-timestep KPIs and their first-order differences. '
            'Raw scores are smoothed with EMA (W=5) and thresholded for alerting. '
            'Corresponds to Section III of Hossain & Aljoby (ICC 2026). '
            'No clause-level output capability. No intent conditioning.'
        ),
        'detection': detection,
        'macro_f1': None,
        'per_clause': None,
    }
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {args.output}")


if __name__ == '__main__':
    main()
