
import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def extract_intent_thresholds(sample):
    intent = sample.get('intent', {})
    perf = intent.get('performance_constraints', {})
    energy = intent.get('energy_constraints', {})
    path = intent.get('path_constraints', {})
    
    return {
        'perf': {
            'delay_threshold_ms': perf.get('delay_threshold_ms'),
            'loss_threshold': perf.get('loss_threshold'),
            'bandwidth_threshold_mbps': perf.get('bandwidth_threshold_mbps'),
            'jitter_threshold_ms': perf.get('jitter_threshold_ms'),
        },
        'energy': {
            'max_power_watts': energy.get('max_power_watts'),
            'min_efficiency_mbps_per_w': energy.get('min_efficiency_mbps_per_w'),
        },
        'path': {
            'waypoints': path.get('waypoints', []),
            'avoid_nodes': path.get('avoid_nodes', []),
            'max_hops': path.get('max_hops'),
        },
    }


def extract_current_observations(sample):
    window = sample.get('window', [])
    if not window:
        return None
    
    last_snapshot = window[-1]
    

    paths = last_snapshot.get('paths', {})

    intent = sample.get('intent', {})
    match = intent.get('match', {})
    intent_src = match.get('src', '')
    intent_dst = match.get('dst', '')

    matched_path = None
    for path_id, path_data in paths.items():
        if isinstance(path_data, dict):
            src = path_data.get('src_host', '')
            dst = path_data.get('dst_host', '')
            if src == intent_src and dst == intent_dst:
                matched_path = path_data
                break

    if matched_path is None and paths:
        first_key = next(iter(paths))
        matched_path = paths[first_key] if isinstance(paths[first_key], dict) else None

    e2e_delay = None
    e2e_loss = None
    e2e_throughput = None
    e2e_jitter = None
    path_nodes = []
    num_hops = 0
    
    if matched_path:
        e2e_delay = matched_path.get('e2e_delay_ms')
        e2e_loss = matched_path.get('e2e_loss_rate')
        e2e_throughput = matched_path.get('e2e_throughput_mbps')
        e2e_jitter = matched_path.get('e2e_jitter_ms')
        path_nodes = matched_path.get('path_nodes', [])
        num_hops = matched_path.get('num_hops', len(path_nodes) - 1 if path_nodes else 0)

    energy_metrics = last_snapshot.get('energy_metrics', {})
    network_metrics = last_snapshot.get('network', {})
    
    total_power = (energy_metrics.get('total_network_power') or
                   network_metrics.get('total_power_watts') or
                   last_snapshot.get('total_power_watts'))
    
    energy_efficiency = (energy_metrics.get('energy_efficiency') or
                         network_metrics.get('energy_efficiency') or
                         last_snapshot.get('energy_efficiency'))
    
    return {
        'e2e_delay_ms': e2e_delay,
        'e2e_loss_rate': e2e_loss,
        'e2e_throughput_mbps': e2e_throughput,
        'e2e_jitter_ms': e2e_jitter,
        'path_nodes': path_nodes,
        'num_hops': num_hops,
        'total_power_watts': total_power,
        'energy_efficiency': energy_efficiency,
    }


def rule_based_predict(thresholds, observations):
    preds = {'perf': 0, 'path': 0, 'energy': 0}
    
    if observations is None:
        return preds

    perf_th = thresholds['perf']

    if (perf_th.get('delay_threshold_ms') is not None and
            observations.get('e2e_delay_ms') is not None):
        if observations['e2e_delay_ms'] > perf_th['delay_threshold_ms']:
            preds['perf'] = 1

    if (perf_th.get('loss_threshold') is not None and
            observations.get('e2e_loss_rate') is not None):
        if observations['e2e_loss_rate'] > perf_th['loss_threshold']:
            preds['perf'] = 1

    if (perf_th.get('bandwidth_threshold_mbps') is not None and
            observations.get('e2e_throughput_mbps') is not None):
        if observations['e2e_throughput_mbps'] < perf_th['bandwidth_threshold_mbps']:
            preds['perf'] = 1

    if (perf_th.get('jitter_threshold_ms') is not None and
            observations.get('e2e_jitter_ms') is not None):
        if observations['e2e_jitter_ms'] > perf_th['jitter_threshold_ms']:
            preds['perf'] = 1
    

    path_th = thresholds['path']
    path_nodes = observations.get('path_nodes', [])
    

    waypoints = path_th.get('waypoints', [])
    if waypoints and path_nodes:
        for wp in waypoints:
            if wp not in path_nodes:
                preds['path'] = 1
                break
    

    avoid_nodes = path_th.get('avoid_nodes', [])
    if avoid_nodes and path_nodes:
        for an in avoid_nodes:
            if an in path_nodes:
                preds['path'] = 1
                break

    if path_th.get('max_hops') is not None and observations.get('num_hops', 0) > 0:
        if observations['num_hops'] > path_th['max_hops']:
            preds['path'] = 1
    

    energy_th = thresholds['energy']
    
    # power: operational > target → 违约
    if (energy_th.get('max_power_watts') is not None and
            observations.get('total_power_watts') is not None):
        if observations['total_power_watts'] > energy_th['max_power_watts']:
            preds['energy'] = 1
    
    # efficiency: operational < target → 违约
    if (energy_th.get('min_efficiency_mbps_per_w') is not None and
            observations.get('energy_efficiency') is not None):
        if observations['energy_efficiency'] < energy_th['min_efficiency_mbps_per_w']:
            preds['energy'] = 1
    
    return preds



CLAUSE_NAMES = ['perf', 'path', 'energy']


def compute_metrics(all_preds, all_labels, total):
    per_clause = {}
    for name in CLAUSE_NAMES:
        p = np.asarray(all_preds[name])
        l = np.asarray(all_labels[name])
        tp = ((p == 1) & (l == 1)).sum()
        fp = ((p == 1) & (l == 0)).sum()
        fn = ((p == 0) & (l == 1)).sum()
        tn = ((p == 0) & (l == 0)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(len(p), 1)
        per_clause[name] = {
            'precision': float(precision), 'recall': float(recall),
            'f1': float(f1), 'accuracy': float(accuracy),
            'support_pos': int((l == 1).sum()),
            'support_neg': int((l == 0).sum()),
        }
    
    macro_f1 = float(np.mean([per_clause[n]['f1'] for n in CLAUSE_NAMES]))
    mean_acc = float(np.mean([per_clause[n]['accuracy'] for n in CLAUSE_NAMES]))
    
    # Overall detection
    any_pred = np.zeros(total, dtype=int)
    any_label = np.zeros(total, dtype=int)
    for name in CLAUSE_NAMES:
        any_pred |= np.asarray(all_preds[name]).astype(int)
        any_label |= np.asarray(all_labels[name]).astype(int)
    
    tp = ((any_pred == 1) & (any_label == 1)).sum()
    fp = ((any_pred == 1) & (any_label == 0)).sum()
    fn = ((any_pred == 0) & (any_label == 1)).sum()
    tn = ((any_pred == 0) & (any_label == 0)).sum()
    det_precision = tp / max(tp + fp, 1)
    det_recall = tp / max(tp + fn, 1)
    det_f1 = 2 * det_precision * det_recall / max(det_precision + det_recall, 1e-8)
    det_acc = (tp + tn) / max(total, 1)
    
    return {
        'macro_f1': macro_f1,
        'mean_accuracy': mean_acc,
        'per_clause': per_clause,
        'detection': {
            'accuracy': float(det_acc),
            'precision': float(det_precision),
            'recall': float(det_recall),
            'f1': float(det_f1),
            'pos_samples': int(any_label.sum()),
            'neg_samples': int(total - any_label.sum()),
        },
    }



def evaluate_rule_based(test_path):

    print(f"Loading test data: {test_path}")
    with open(test_path, 'r') as f:
        data = json.load(f)
    print(f"  Total samples: {len(data)}")
    
    all_preds = {name: [] for name in CLAUSE_NAMES}
    all_labels = {name: [] for name in CLAUSE_NAMES}

    n_no_obs = 0
    n_no_threshold = 0
    
    for i, sample in enumerate(data):
        thresholds = extract_intent_thresholds(sample)

        observations = extract_current_observations(sample)
        
        if observations is None:
            n_no_obs += 1

        preds = rule_based_predict(thresholds, observations)

        fcl = sample.get('future_clause_labels', {})
        if not fcl:
            fcl = sample.get('current_clause_labels', {})
        
        for name in CLAUSE_NAMES:
            all_preds[name].append(preds[name])
            all_labels[name].append(int(fcl.get(name, 0)))
    
    total = len(data)
    print(f"  Samples with no observations: {n_no_obs}")

    metrics = compute_metrics(all_preds, all_labels, total)
    
    return metrics


def print_results(metrics, method_name="Rule-based [Dzeparoska et al.]"):
    print()
    print("=" * 60)
    print(f"  {method_name}")
    print("=" * 60)
    
    print(f"\n  Macro F1:           {metrics['macro_f1']:.4f}")
    print(f"  Mean accuracy:      {metrics['mean_accuracy']:.4f}")
    
    print(f"\n  Per-clause results:")
    print(f"  {'Clause':<14} {'Prec':>6} {'Recall':>8} {'F1':>8} {'Acc':>8} {'Pos':>6} {'Neg':>6}")
    print(f"  {'-'*60}")
    for name in CLAUSE_NAMES:
        c = metrics['per_clause'][name]
        print(f"  {name:<14} {c['precision']:>6.3f} {c['recall']:>8.3f} "
              f"{c['f1']:>8.3f} {c['accuracy']:>8.3f} "
              f"{c['support_pos']:>6d} {c['support_neg']:>6d}")
    
    det = metrics['detection']
    print(f"\n  Overall drift detection (any clause):")
    print(f"    Precision: {det['precision']:.4f}")
    print(f"    Recall:    {det['recall']:.4f}")
    print(f"    F1:        {det['f1']:.4f}")
    print(f"    Accuracy:  {det['accuracy']:.4f}")
    print(f"    Pos/Neg:   {det['pos_samples']}/{det['neg_samples']}")


def main():
    parser = argparse.ArgumentParser(description='Rule-based thresholding baseline')
    parser.add_argument('--test', type=str, default=r'D:\datasets\Intent_drift\icsme\test.json',
                        help='Path to test data JSON')
    parser.add_argument('--output', type=str, default=r'D:\pycharmProjects\IntentDrift\checkpoints1\results_rule_based.json',
                        help='Output path for results JSON')
    args = parser.parse_args()
    
    t0 = time.time()
    metrics = evaluate_rule_based(args.test)
    t1 = time.time()
    
    print_results(metrics)
    print(f"\n  Evaluation time: {t1 - t0:.1f}s")

    output = {
        'method': 'Rule-based Thresholding [Dzeparoska et al. NOMS 2024, 2025]',
        'test_data': args.test,
        'description': (
            'Reactive baseline: compares current-timestep KPIs against intent thresholds. '
            'No temporal modeling, no proactive prediction. '
            'Corresponds to the KPI deviation detection in Dzeparoska et al.'
        ),
        'metrics': metrics,
    }
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {args.output}")


if __name__ == '__main__':
    main()
