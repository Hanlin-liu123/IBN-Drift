#!/usr/bin/env python3
"""
baseline_safla.py

SAFLA-style Baseline (对应 Kou et al., "SAFLA: Semantic-Aware Full Lifecycle
Assurance for Intent-Driven Networks", IEEE TCCN 2026)

三个变种：
  1. SAFLA-vanilla:  原论文公式(5)-(7)，只比较 (src, dst, protocol) 集合
  2. SAFLA-extended: 扩展到 (src, dst, protocol, path) 比对，能检测路径变更
  3. SAFLA-full:    集合比对 + QoS 阈值检测（原论文公式 2-4），覆盖 perf+path

所有变种都是 reactive 的——只看当前时刻的状态，不做时序预测。
SAFLA 不具备 energy clause 检测能力（原论文不涉及能耗约束）。

用法:
    python baseline_safla.py --test data/real_trace_dataset/test.json \
                             --output checkpoints/results_safla.json
"""
import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLAUSE_NAMES = ['perf', 'path', 'energy']


# ============================================================
# SAFLA 核心：意图集合比对 (论文 Section IV-B, Algorithm 1)
# ============================================================

def extract_declared_intent(sample):
    """
    提取用户声明的意图 I = {(src, dst, protocol)}
    以及声明的路径（用于 extended 变种）
    """
    intent = sample.get('intent', {})
    match = intent.get('match', {})
    
    src = match.get('src', '')
    dst = match.get('dst', '')
    protocol = match.get('protocol', 'UDP')
    
    # 声明的路径：从 baseline_routing_paths 中获取
    baseline_routing = sample.get('baseline_routing_paths', {})
    flow_key = f"{src}-{dst}"
    declared_path = baseline_routing.get(flow_key, [])
    
    return {
        'src': src,
        'dst': dst,
        'protocol': protocol,
        'declared_path': tuple(declared_path) if declared_path else (),
    }


def extract_observed_intent(sample):
    """
    从当前网络观测中提取 "已部署的意图" Î
    
    对应 SAFLA 论文 Section IV-A 的 Intent Extraction：
    从流表/网络观测中推断当前实际部署的意图。
    在我们的数据中，这对应于观测到的实际路径和流量状态。
    """
    window = sample.get('window', [])
    if not window:
        return None
    
    last_snapshot = window[-1]
    paths = last_snapshot.get('paths', {})
    
    intent = sample.get('intent', {})
    match = intent.get('match', {})
    intent_src = match.get('src', '')
    intent_dst = match.get('dst', '')
    
    # 找匹配的路径
    observed_path = []
    has_traffic = False
    e2e_delay = None
    e2e_loss = None
    e2e_throughput = None
    
    for pid, pdata in paths.items():
        if isinstance(pdata, dict):
            src = pdata.get('src_host', '')
            dst = pdata.get('dst_host', '')
            if src == intent_src and dst == intent_dst:
                observed_path = pdata.get('path_nodes', [])
                e2e_throughput = pdata.get('e2e_throughput_mbps', 0)
                e2e_delay = pdata.get('e2e_delay_ms', 0)
                e2e_loss = pdata.get('e2e_loss_rate', 0)
                has_traffic = (e2e_throughput or 0) > 0.001
                break
    
    # 如果没找到精确匹配，取第一条路径
    if not observed_path and paths:
        first_key = next(iter(paths))
        pdata = paths[first_key]
        if isinstance(pdata, dict):
            observed_path = pdata.get('path_nodes', [])
            e2e_throughput = pdata.get('e2e_throughput_mbps', 0)
            e2e_delay = pdata.get('e2e_delay_ms', 0)
            e2e_loss = pdata.get('e2e_loss_rate', 0)
            has_traffic = (e2e_throughput or 0) > 0.001
    
    return {
        'src': intent_src,
        'dst': intent_dst,
        'protocol': match.get('protocol', 'UDP'),
        'observed_path': tuple(observed_path) if observed_path else (),
        'has_traffic': has_traffic,
        'e2e_delay': e2e_delay,
        'e2e_loss': e2e_loss,
        'e2e_throughput': e2e_throughput,
    }


def safla_vanilla_predict(declared, observed):
    """
    SAFLA-vanilla: 原论文公式 (5)-(7)
    
    只比较 (src, dst, protocol) 是否一致。
    
    χ(i→j) = 1 if 声明的意图在网络中不存在 (公式 5)
    γ(j→i) = 1 if 网络中有未声明的意图 (公式 6)
    ΔS = Σχ + Σγ (公式 7)
    
    在我们的场景中：只有一条意图，所以：
    - 如果观测到的流量存在且匹配声明 → ΔS = 0
    - 如果观测不到流量 → χ = 1 (声明了但没部署)
    """
    preds = {'perf': 0, 'path': 0, 'energy': 0}
    
    if observed is None:
        preds['path'] = 1  # 无法观测 → 视为意图未部署
        return preds
    
    # 公式 (5): 声明的意图是否被部署？
    declared_key = (declared['src'], declared['dst'], declared['protocol'])
    observed_key = (observed['src'], observed['dst'], observed['protocol'])
    
    if declared_key != observed_key or not observed['has_traffic']:
        preds['path'] = 1  # 意图未被部署或无流量
    
    # Vanilla 版本不检测 perf 和 energy
    # 也不检测路径是否变化（只看端点是否匹配）
    
    return preds


def safla_extended_predict(declared, observed):
    """
    SAFLA-extended: 扩展到 (src, dst, protocol, path) 比对
    
    在 vanilla 基础上，进一步检查实际路径是否与声明路径一致。
    这对应 SAFLA 论文中 "comparing extracted intents with original 
    user-defined intents" 的扩展版——不仅比较端点，还比较路径。
    """
    preds = {'perf': 0, 'path': 0, 'energy': 0}
    
    if observed is None:
        preds['path'] = 1
        return preds
    
    # 端点检查（和 vanilla 一样）
    declared_key = (declared['src'], declared['dst'], declared['protocol'])
    observed_key = (observed['src'], observed['dst'], observed['protocol'])
    
    if declared_key != observed_key or not observed['has_traffic']:
        preds['path'] = 1
        return preds
    
    # 路径检查：声明路径和观测路径是否一致
    if declared['declared_path'] and observed['observed_path']:
        if declared['declared_path'] != observed['observed_path']:
            preds['path'] = 1
    
    # Extended 版本仍不检测 perf 和 energy
    
    return preds


def safla_full_predict(declared, observed, thresholds):
    """
    SAFLA-full: 集合比对 + QoS 阈值检测
    
    对应 SAFLA 论文的完整框架：
    - 语义漂移检测: 公式 (5)-(7)，比较 (src, dst, protocol, path)
    - 性能漂移检测: 公式 (2)-(4)，比较 operational KPI vs target KPI
      δk = p_mon_k - p_req_k, q(δk) = 1 if |δk| ≥ λk
    
    这是 SAFLA 能达到的最强配置——同时覆盖配置一致性和性能合规。
    但仍然不覆盖 energy clause（SAFLA 论文不涉及能耗约束）。
    """
    preds = {'perf': 0, 'path': 0, 'energy': 0}
    
    if observed is None:
        preds['path'] = 1
        return preds
    
    # === 语义漂移检测 (和 extended 一样) ===
    declared_key = (declared['src'], declared['dst'], declared['protocol'])
    observed_key = (observed['src'], observed['dst'], observed['protocol'])
    
    if declared_key != observed_key or not observed['has_traffic']:
        preds['path'] = 1
    
    if declared['declared_path'] and observed['observed_path']:
        if declared['declared_path'] != observed['observed_path']:
            preds['path'] = 1
    
    # === 性能漂移检测 (公式 2-4) ===
    # δk = p_mon_k - p_req_k
    # q(δk) = 1 if |δk| ≥ λk
    perf_th = thresholds.get('perf', {})
    
    # delay: operational > target → performance drift
    if (perf_th.get('delay_threshold_ms') is not None and
            observed.get('e2e_delay') is not None):
        if observed['e2e_delay'] > perf_th['delay_threshold_ms']:
            preds['perf'] = 1
    
    # loss: operational > target → performance drift
    if (perf_th.get('loss_threshold') is not None and
            observed.get('e2e_loss') is not None):
        if observed['e2e_loss'] > perf_th['loss_threshold']:
            preds['perf'] = 1
    
    # throughput: operational < target → performance drift
    if (perf_th.get('bandwidth_threshold_mbps') is not None and
            observed.get('e2e_throughput') is not None):
        if observed['e2e_throughput'] < perf_th['bandwidth_threshold_mbps']:
            preds['perf'] = 1
    
    # 路径不可用也触发 path（和 auto_label 一致）
    if (perf_th.get('delay_threshold_ms') is not None and
            observed.get('e2e_delay') is not None):
        if observed['e2e_delay'] > perf_th['delay_threshold_ms'] * 5:
            preds['path'] = 1
    
    if observed.get('e2e_loss') is not None and observed['e2e_loss'] > 0.5:
        preds['path'] = 1
    
    # SAFLA 不检测 energy — 始终为 0
    
    return preds


def extract_intent_thresholds(sample):
    """提取意图的 QoS 阈值（用于 SAFLA-full 的性能漂移检测）"""
    intent = sample.get('intent', {})
    perf = intent.get('performance_constraints', {})
    
    return {
        'perf': {
            'delay_threshold_ms': perf.get('delay_threshold_ms'),
            'loss_threshold': perf.get('loss_threshold'),
            'bandwidth_threshold_mbps': perf.get('bandwidth_threshold_mbps'),
            'jitter_threshold_ms': perf.get('jitter_threshold_ms'),
        },
    }


# ============================================================
# 评估指标
# ============================================================

def compute_metrics(all_preds, all_labels, total):
    """和 CLASP 用完全相同的指标"""
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
    
    return {
        'macro_f1': macro_f1,
        'per_clause': per_clause,
        'detection': {
            'precision': float(det_precision),
            'recall': float(det_recall),
            'f1': float(det_f1),
            'pos_samples': int(any_label.sum()),
            'neg_samples': int(total - any_label.sum()),
        },
    }


# ============================================================
# 主评估逻辑
# ============================================================

def evaluate_safla(test_path, variant='full'):
    """
    在 test set 上评估 SAFLA baseline
    
    Args:
        variant: 'vanilla' / 'extended' / 'full'
    """
    with open(test_path, 'r') as f:
        data = json.load(f)
    
    all_preds = {name: [] for name in CLAUSE_NAMES}
    all_labels = {name: [] for name in CLAUSE_NAMES}
    
    n_no_baseline_routing = 0
    
    for sample in data:
        # 提取声明意图和观测意图
        declared = extract_declared_intent(sample)
        observed = extract_observed_intent(sample)
        
        if not declared.get('declared_path'):
            n_no_baseline_routing += 1
        
        # 根据变种选择预测函数
        if variant == 'vanilla':
            preds = safla_vanilla_predict(declared, observed)
        elif variant == 'extended':
            preds = safla_extended_predict(declared, observed)
        elif variant == 'full':
            thresholds = extract_intent_thresholds(sample)
            preds = safla_full_predict(declared, observed, thresholds)
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        # Ground truth
        fcl = sample.get('future_clause_labels', {})
        if not fcl:
            fcl = sample.get('current_clause_labels', {})
        
        for name in CLAUSE_NAMES:
            all_preds[name].append(preds[name])
            all_labels[name].append(int(fcl.get(name, 0)))
    
    total = len(data)
    metrics = compute_metrics(all_preds, all_labels, total)
    
    return metrics, n_no_baseline_routing


def print_results(metrics, variant_name):
    print(f"\n  {variant_name}")
    print(f"  {'='*55}")
    print(f"\n  Detection F1:  {metrics['detection']['f1']:.4f}  "
          f"(P={metrics['detection']['precision']:.4f} "
          f"R={metrics['detection']['recall']:.4f})")
    print(f"  Macro F1:      {metrics['macro_f1']:.4f}")
    
    print(f"\n  Per-clause:")
    for name in CLAUSE_NAMES:
        c = metrics['per_clause'][name]
        note = ""
        if variant_name.startswith("SAFLA-vanilla") and name in ['perf', 'energy']:
            note = " (not detected by this variant)"
        elif variant_name.startswith("SAFLA-extended") and name in ['perf', 'energy']:
            note = " (not detected by this variant)"
        elif "SAFLA" in variant_name and name == 'energy':
            note = " (not in SAFLA's scope)"
        print(f"    {name:<8}: P={c['precision']:.3f} R={c['recall']:.3f} "
              f"F1={c['f1']:.3f}{note}")


def main():
    parser = argparse.ArgumentParser(description='SAFLA-style baseline')
    parser.add_argument('--test', type=str, default=r'D:\datasets\Intent_drift\icsme\test.json')
    parser.add_argument('--output', type=str, default='checkpoints/results_safla.json')
    args = parser.parse_args()
    
    print(f"Loading test data: {args.test}")
    with open(args.test, 'r') as f:
        data = json.load(f)
    print(f"  Test samples: {len(data)}")
    
    # 检查 baseline_routing_paths 是否存在
    has_brp = sum(1 for s in data if s.get('baseline_routing_paths'))
    print(f"  Samples with baseline_routing_paths: {has_brp}/{len(data)}")
    
    results = {}
    
    # ============================================================
    # SAFLA-vanilla: 只比较 (src, dst, protocol)
    # ============================================================
    print("\n" + "=" * 60)
    print("SAFLA-vanilla [Kou et al., Eq. 5-7]")
    print("=" * 60)
    
    t0 = time.time()
    m_vanilla, n_no_brp = evaluate_safla(args.test, variant='vanilla')
    t1 = time.time()
    
    print_results(m_vanilla, "SAFLA-vanilla (src,dst,proto match only)")
    print(f"\n  Time: {t1-t0:.1f}s")
    results['vanilla'] = m_vanilla
    
    # ============================================================
    # SAFLA-extended: 比较 (src, dst, protocol, path)
    # ============================================================
    print("\n" + "=" * 60)
    print("SAFLA-extended [Kou et al., extended with path comparison]")
    print("=" * 60)
    
    t0 = time.time()
    m_extended, _ = evaluate_safla(args.test, variant='extended')
    t1 = time.time()
    
    print_results(m_extended, "SAFLA-extended (+ path comparison)")
    print(f"\n  Time: {t1-t0:.1f}s")
    if n_no_brp > 0:
        print(f"  Warning: {n_no_brp} samples missing baseline_routing_paths")
    results['extended'] = m_extended
    
    # ============================================================
    # SAFLA-full: 集合比对 + QoS 阈值检测
    # ============================================================
    print("\n" + "=" * 60)
    print("SAFLA-full [Kou et al., Eq. 2-7, semantic + performance drift]")
    print("=" * 60)
    
    t0 = time.time()
    m_full, _ = evaluate_safla(args.test, variant='full')
    t1 = time.time()
    
    print_results(m_full, "SAFLA-full (+ QoS threshold check)")
    print(f"\n  Time: {t1-t0:.1f}s")
    results['full'] = m_full
    
    # ============================================================
    # 三个变种的对比总结
    # ============================================================
    print("\n" + "=" * 60)
    print("Summary: SAFLA variants comparison")
    print("=" * 60)
    print(f"\n  {'Variant':<25} {'Det.F1':>8} {'Macro':>8} {'perf':>8} {'path':>8} {'energy':>8}")
    print(f"  {'-'*65}")
    for vname, vkey in [('SAFLA-vanilla', 'vanilla'),
                         ('SAFLA-extended', 'extended'),
                         ('SAFLA-full', 'full')]:
        m = results[vkey]
        print(f"  {vname:<25} {m['detection']['f1']:>8.3f} {m['macro_f1']:>8.3f} "
              f"{m['per_clause']['perf']['f1']:>8.3f} "
              f"{m['per_clause']['path']['f1']:>8.3f} "
              f"{m['per_clause']['energy']['f1']:>8.3f}")
    
    # 保存
    output = {
        'method': 'SAFLA-style baselines [Kou et al., IEEE TCCN 2026]',
        'test_data': args.test,
        'description': {
            'vanilla': (
                'Original SAFLA: compare declared vs deployed intent tuples '
                '(src, dst, protocol). Corresponds to Eq. (5)-(7). '
                'Cannot detect performance or energy drift.'
            ),
            'extended': (
                'Extended SAFLA: additionally compare declared vs observed '
                'forwarding paths. Can detect path changes but not QoS or energy drift.'
            ),
            'full': (
                'Full SAFLA: semantic drift detection (path comparison) + '
                'performance drift detection (QoS threshold comparison, Eq. 2-4). '
                'Still cannot detect energy drift (outside SAFLA scope).'
            ),
        },
        'results': {
            'vanilla': results['vanilla'],
            'extended': results['extended'],
            'full': results['full'],
        },
    }
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {args.output}")


if __name__ == '__main__':
    main()
