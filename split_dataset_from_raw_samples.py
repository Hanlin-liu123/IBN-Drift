

import os
import json
import argparse
import random
from collections import defaultdict



RAW_FILE = r'D:\shareWithUbuntu\samples_raw_abilene.jsonl'
OUTPUT_DIR = r'D:\datasets\Intent_drift\abilene'

MIN_PERSIST = 2
MAX_GAP_SECONDS = 12.0
RANDOM_SEED = 42



def build_sequences(samples, window_size=10, horizon=3, min_persist=2,
                    max_gap_seconds=12.0):

    groups = defaultdict(list)

    for s in samples:
        key = (
            s.get('topology', ''),
            s.get('routing_id', 0),
            s.get('scheduling_config', 0),
            s.get('intent', {}).get('intent_id', ''),
        )
        groups[key].append(s)

    sequence_samples = []
    n_filtered_gap = 0

    for group_key, group_samples in groups.items():
        group_samples.sort(key=lambda x: x.get('timestamp', 0))
        n = len(group_samples)

        if n < window_size + horizon:
            continue

        for i in range(n - window_size - horizon + 1):
            window = group_samples[i: i + window_size]
            future = group_samples[i + window_size: i + window_size + horizon]

            # 时间断点过滤
            full_seq = window + future
            has_gap = False

            for k in range(1, len(full_seq)):
                gap = full_seq[k].get('timestamp', 0) - full_seq[k - 1].get('timestamp', 0)
                if gap > max_gap_seconds or gap < 0:
                    has_gap = True
                    break

            if has_gap:
                n_filtered_gap += 1
                continue

            # Persistence-aware future labels
            future_clause_labels = {'perf': 0, 'path': 0, 'energy': 0}

            for clause_name in ['perf', 'path', 'energy']:
                consecutive = 0

                for f in future:
                    clause_labels = f.get('clause_labels', {})
                    cl = clause_labels.get(clause_name, 0)

                    if cl > 0:
                        consecutive += 1
                        if consecutive >= min_persist:
                            future_clause_labels[clause_name] = 1
                            break
                    else:
                        consecutive = 0

            future_has_any_drift = any(future_clause_labels.values())


            if not future_has_any_drift:
                future_label = 0
            elif future_clause_labels['path']:
                future_label = 2
            elif future_clause_labels['perf']:
                future_label = 1
            elif future_clause_labels['energy']:
                future_label = 3
            else:
                future_label = 0

            current = window[-1]

            topology, routing_id, scheduling_config, intent_id = group_key

            seq_sample = {
                'window': window,
                'intent': current.get('intent', {}),
                'future_clause_labels': future_clause_labels,
                'future_has_any_drift': future_has_any_drift,
                'future_label': future_label,
                'current_clause_labels': current.get(
                    'clause_labels',
                    {'perf': 0, 'path': 0, 'energy': 0}
                ),

                'experiment_id': f"{topology}_{routing_id}_{scheduling_config}_{intent_id}",
                'intent_id': intent_id,

                'window_start_ts': window[0].get('timestamp', 0),
                'window_end_ts': window[-1].get('timestamp', 0),
                'drift_location': None,
                'baseline_routing_paths': window[0].get('baseline_routing_paths', {}),
            }


            if future_has_any_drift:
                for f in future:
                    cl = f.get('clause_labels', {})
                    if any(cl.values()) and f.get('drift_location'):
                        seq_sample['drift_location'] = f['drift_location']
                        break

                if not seq_sample['drift_location']:
                    for f in reversed(window):
                        cl = f.get('clause_labels', {})
                        if any(cl.values()) and f.get('drift_location'):
                            seq_sample['drift_location'] = f['drift_location']
                            break

            sequence_samples.append(seq_sample)

    if n_filtered_gap > 0:
        print(
            f"  Filtered {n_filtered_gap} windows due to time gaps "
            f"(>{max_gap_seconds}s between adjacent snapshots)"
        )

    return sequence_samples




def is_drift_sample(sample):
    labels = sample.get('future_clause_labels', {})
    return any(labels.get(k, 0) for k in ['perf', 'path', 'energy'])


def count_drift(samples):
    return sum(1 for s in samples if is_drift_sample(s))


def drift_ratio(samples):
    return count_drift(samples) / max(len(samples), 1)


def clause_ratios(samples):
    result = {}
    for k in ['perf', 'path', 'energy']:
        result[k] = sum(
            1 for s in samples
            if s.get('future_clause_labels', {}).get(k, 0)
        ) / max(len(samples), 1)
    return result


# ============================================================
# Group-level stratified split
# ============================================================

def split_and_save(all_seq_samples, output_dir, window_size, horizon, suffix=''):

    os.makedirs(output_dir, exist_ok=True)

    rng = random.Random(RANDOM_SEED)

    seq_groups = defaultdict(list)
    for s in all_seq_samples:
        seq_groups[s['experiment_id']].append(s)

    total_samples = len(all_seq_samples)
    total_drift = count_drift(all_seq_samples)
    global_drift_ratio = total_drift / max(total_samples, 1)

    print(
        f"  Group-level stratified split: {len(seq_groups)} groups, "
        f"{total_samples} samples, global drift={global_drift_ratio:.1%}"
    )

    group_stats = []

    for exp_id, samples in seq_groups.items():
        samples.sort(key=lambda x: x.get('window_end_ts', 0))

        n = len(samples)
        n_drift = count_drift(samples)
        ratio = n_drift / max(n, 1)

        group_stats.append({
            'exp_id': exp_id,
            'samples': samples,
            'n': n,
            'n_drift': n_drift,
            'drift_ratio': ratio,
        })

    target_counts = {
        'train': int(total_samples * 0.70),
        'val': int(total_samples * 0.15),
        'test': total_samples - int(total_samples * 0.70) - int(total_samples * 0.15),
    }

    splits = {
        'train': [],
        'val': [],
        'test': [],
    }

    split_stats = {
        'train': {'n': 0, 'n_drift': 0},
        'val': {'n': 0, 'n_drift': 0},
        'test': {'n': 0, 'n_drift': 0},
    }

    rng.shuffle(group_stats)
    group_stats.sort(
        key=lambda g: (g['n'], abs(g['drift_ratio'] - global_drift_ratio)),
        reverse=True
    )

    def projected_cost(split_name, group):
        cur_n = split_stats[split_name]['n']
        cur_d = split_stats[split_name]['n_drift']

        new_n = cur_n + group['n']
        new_d = cur_d + group['n_drift']

        target_n = target_counts[split_name]

        size_cost = abs(new_n - target_n) / max(target_n, 1)

        new_ratio = new_d / max(new_n, 1)
        drift_cost = abs(new_ratio - global_drift_ratio)

        return size_cost + 2.0 * drift_cost

    for group in group_stats:
        candidate_splits = []

        for split_name in ['train', 'val', 'test']:
            current_n = split_stats[split_name]['n']
            target_n = target_counts[split_name]

            if current_n < target_n * 1.10:
                candidate_splits.append(split_name)

        if not candidate_splits:
            candidate_splits = ['train', 'val', 'test']

        best_split = min(candidate_splits, key=lambda sp: projected_cost(sp, group))

        splits[best_split].extend(group['samples'])
        split_stats[best_split]['n'] += group['n']
        split_stats[best_split]['n_drift'] += group['n_drift']

    train_samples = splits['train']
    val_samples = splits['val']
    test_samples = splits['test']

    for data in [train_samples, val_samples, test_samples]:
        data.sort(key=lambda x: (x.get('experiment_id', ''), x.get('window_end_ts', 0)))

    train_ids = {s['experiment_id'] for s in train_samples}
    val_ids = {s['experiment_id'] for s in val_samples}
    test_ids = {s['experiment_id'] for s in test_samples}

    assert train_ids.isdisjoint(val_ids), "Leakage: train and val share experiment_id"
    assert train_ids.isdisjoint(test_ids), "Leakage: train and test share experiment_id"
    assert val_ids.isdisjoint(test_ids), "Leakage: val and test share experiment_id"

    print("  Split summary:")
    for name, data in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
        n = len(data)
        d = count_drift(data)
        cr = clause_ratios(data)
        print(
            f"    {name}: {n} samples, drift={d} ({d / max(n, 1):.1%}), "
            f"perf={cr['perf']:.1%}, path={cr['path']:.1%}, energy={cr['energy']:.1%}, "
            f"target_size={target_counts[name]}"
        )

    for name, data in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
        path = os.path.join(output_dir, f'{name}{suffix}.json')

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, default=str)

        n_drift = count_drift(data)
        print(
            f"  Saved {path}: {len(data)} samples, "
            f"drift={n_drift} ({n_drift / max(len(data), 1):.1%})"
        )

    meta_path = os.path.join(output_dir, f'split_meta{suffix}.json')
    split_meta = {
        'window_size': window_size,
        'horizon': horizon,
        'min_persist': MIN_PERSIST,
        'max_gap_seconds': MAX_GAP_SECONDS,
        'random_seed': RANDOM_SEED,
        'split_strategy': 'group_level_stratified_by_experiment_id',
        'experiment_id_definition': 'topology_routing_id_scheduling_config_intent_id',
        'global': {
            'n_samples': total_samples,
            'n_drift': total_drift,
            'drift_ratio': global_drift_ratio,
        },
        'splits': {
            'train': {
                'n_samples': len(train_samples),
                'n_drift': count_drift(train_samples),
                'drift_ratio': drift_ratio(train_samples),
                'experiment_ids': sorted(list(train_ids)),
            },
            'val': {
                'n_samples': len(val_samples),
                'n_drift': count_drift(val_samples),
                'drift_ratio': drift_ratio(val_samples),
                'experiment_ids': sorted(list(val_ids)),
            },
            'test': {
                'n_samples': len(test_samples),
                'n_drift': count_drift(test_samples),
                'drift_ratio': drift_ratio(test_samples),
                'experiment_ids': sorted(list(test_ids)),
            },
        }
    }

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(split_meta, f, indent=2, default=str)

    print(f"  Saved split metadata: {meta_path}")



def load_raw_samples(raw_path):
    print(f"Loading raw snapshots: {raw_path}")

    samples = []

    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    print(f"  Loaded {len(samples)} raw snapshots")
    return samples




def rebuild_one(raw_samples, output_dir, window_size, horizon, suffix):
    print(f"\n{'=' * 60}")
    print(f"  T={window_size}, h={horizon}, suffix='{suffix}'")
    print(f"{'=' * 60}")

    print(
        f"  Building sequences "
        f"(T={window_size}, h={horizon}, persist={MIN_PERSIST}, "
        f"max_gap={MAX_GAP_SECONDS}s)..."
    )

    seq_samples = build_sequences(
        raw_samples,
        window_size=window_size,
        horizon=horizon,
        min_persist=MIN_PERSIST,
        max_gap_seconds=MAX_GAP_SECONDS,
    )

    print(f"  Generated {len(seq_samples)} sequence samples")

    if not seq_samples:
        print(f"  WARNING: No samples generated for T={window_size}, h={horizon}!")
        return

    split_and_save(
        seq_samples,
        output_dir,
        window_size,
        horizon,
        suffix=suffix
    )



def main():
    parser = argparse.ArgumentParser(
        description='Rebuild sequences from raw snapshots with group-level stratified split.'
    )

    parser.add_argument(
        '--raw',
        type=str,
        default=RAW_FILE,
        help='Path to samples_raw_new.jsonl'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=OUTPUT_DIR,
        help='Output directory'
    )

    parser.add_argument(
        '--window-size',
        type=int,
        default=10,
        help='Sliding window size T'
    )

    parser.add_argument(
        '--horizon',
        type=int,
        default=3,
        help='Prediction horizon h'
    )

    parser.add_argument(
        '--suffix',
        type=str,
        default='',
        help='Suffix for output files'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Generate all sensitivity experiment datasets'
    )

    args = parser.parse_args()

    raw_samples = load_raw_samples(args.raw)

    if args.batch:
        # 实验1: 固定 h=3, 变 T
        for T in [5, 8, 10, 12, 15]:
            rebuild_one(
                raw_samples,
                args.output_dir,
                window_size=T,
                horizon=3,
                suffix=f'_T{T}'
            )

        # 实验2: 固定 T=10, 变 h
        for h in [3, 5, 7, 9, 11]:
            rebuild_one(
                raw_samples,
                args.output_dir,
                window_size=10,
                horizon=h,
                suffix=f'_h{h}'
            )

        print(f"\n{'=' * 60}")
        print("  All done! Generated datasets:")
        print("    T experiments: T=5,8,10,12,15 with h=3")
        print("    h experiments: h=3,5,7,9,11 with T=10")
        print(f"  Output directory: {args.output_dir}")
        print(f"{'=' * 60}")

    else:
        suffix = args.suffix or f'_T{args.window_size}_h{args.horizon}_1'
        rebuild_one(
            raw_samples,
            args.output_dir,
            window_size=args.window_size,
            horizon=args.horizon,
            suffix=suffix
        )


if __name__ == '__main__':
    main()