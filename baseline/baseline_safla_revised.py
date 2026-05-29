#!/usr/bin/env python3
"""
baseline_safla_revised.py

A stronger SAFLA-style baseline adapted to clause-level intent-drift datasets.

Compared with the original simplified baseline, this version is closer to Kou et al.
("SAFLA: Semantic-Aware Full Lifecycle Assurance for Intent-Driven Networks",
IEEE TCCN 2026) in the following sense:

1) It extracts an observed intent set \hat{I} from ALL currently observed paths,
   instead of only checking the declared intent.
2) It performs bidirectional semantic comparison:
      - declared but unimplemented intents   (chi)
      - observed but undeclared intents      (gamma)
3) It keeps three practical variants:
      - SAFLA-vanilla : tuple-level semantic comparison on (src,dst,protocol)
      - SAFLA-extended: path-aware semantic comparison on (src,dst,protocol,path)
      - SAFLA-full    : path-aware semantic comparison + QoS threshold checks
4) It explicitly keeps energy clause disabled because SAFLA does not model energy.

IMPORTANT
---------
This is still an *adapted baseline* for a clause-level dataset, not a literal
reproduction of the paper's SDN flow-table clustering/aggregation/linking
pipeline. The original SAFLA extracts intents bottom-up from switch flow-table
configurations. Here, because the dataset exposes path-level observations rather
than raw OpenFlow tables, we approximate \hat{I} using observed end-to-end paths.

Usage:
    python baseline_safla_revised.py \
        --test data/real_trace_dataset/test.json \
        --output checkpoints/results_safla_revised.json \
        --label-source current

If you want to benchmark against future labels in your dataset, you may use:
    --label-source future
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

CLAUSE_NAMES = ["perf", "path", "energy"]
EPS_TRAFFIC = 1e-3


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class IntentTuple:
    src: str
    dst: str
    protocol: str


@dataclass(frozen=True)
class IntentPathTuple:
    src: str
    dst: str
    protocol: str
    path: Tuple[str, ...]


@dataclass
class ObservedIntent:
    src: str
    dst: str
    protocol: str
    path: Tuple[str, ...]
    has_traffic: bool
    e2e_delay: Optional[float]
    e2e_loss: Optional[float]
    e2e_throughput: Optional[float]
    raw_id: str = ""

    @property
    def tuple_key(self) -> IntentTuple:
        return IntentTuple(self.src, self.dst, self.protocol)

    @property
    def path_key(self) -> IntentPathTuple:
        return IntentPathTuple(self.src, self.dst, self.protocol, self.path)


# ============================================================
# Helpers
# ============================================================


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None



def normalize_protocol(p: Any, default: str = "UDP") -> str:
    if p is None:
        return default
    s = str(p).strip()
    return s.upper() if s else default



def normalize_path(path: Any) -> Tuple[str, ...]:
    if not path:
        return ()
    if isinstance(path, (list, tuple)):
        return tuple(str(x) for x in path)
    return (str(path),)



def get_last_snapshot(sample: Dict[str, Any]) -> Dict[str, Any]:
    window = sample.get("window", [])
    if isinstance(window, list) and window:
        last = window[-1]
        if isinstance(last, dict):
            return last
    return {}



def get_label_dict(sample: Dict[str, Any], label_source: str) -> Dict[str, int]:
    if label_source == "future":
        return sample.get("future_clause_labels", {}) or sample.get("current_clause_labels", {}) or {}
    return sample.get("current_clause_labels", {}) or sample.get("future_clause_labels", {}) or {}


# ============================================================
# Declared intent extraction
# ============================================================


def extract_declared_intents(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Dataset-adapted declared intent extraction.

    Current dataset appears to provide a single intent in sample['intent'].
    We still return a list so the logic naturally supports multiple intents.
    """
    intent = sample.get("intent", {}) or {}
    match = intent.get("match", {}) or {}

    src = str(match.get("src", "") or "")
    dst = str(match.get("dst", "") or "")
    protocol = normalize_protocol(match.get("protocol", "UDP"))

    baseline_routing = sample.get("baseline_routing_paths", {}) or {}
    declared_path = ()
    if src and dst:
        flow_key = f"{src}-{dst}"
        declared_path = normalize_path(baseline_routing.get(flow_key, []))

    if not src or not dst:
        return []

    return [{
        "src": src,
        "dst": dst,
        "protocol": protocol,
        "declared_path": declared_path,
    }]


# ============================================================
# Observed intent-set extraction (adapted SAFLA bottom-up)
# ============================================================


def infer_protocol_from_path_record(pdata: Dict[str, Any], sample: Dict[str, Any]) -> str:
    # Prefer explicit protocol in the path record; otherwise fall back to declared intent.
    proto = pdata.get("protocol")
    if proto is not None:
        return normalize_protocol(proto)

    intent = sample.get("intent", {}) or {}
    match = intent.get("match", {}) or {}
    return normalize_protocol(match.get("protocol", "UDP"))



def path_record_has_signal(pdata: Dict[str, Any]) -> bool:
    tput = safe_float(pdata.get("e2e_throughput_mbps"))
    if tput is not None and tput > EPS_TRAFFIC:
        return True
    path_nodes = normalize_path(pdata.get("path_nodes", []))
    return len(path_nodes) > 0



def extract_observed_intents(sample: Dict[str, Any], active_only: bool = True) -> List[ObservedIntent]:
    """
    Approximate SAFLA's extracted intent set \hat{I} from all observed end-to-end
    path records in the current snapshot.

    This is the most important fix relative to the original simplified code:
    we no longer inspect only the declared flow; instead, we construct the whole
    observed intent set and then do bidirectional comparisons.
    """
    last_snapshot = get_last_snapshot(sample)
    paths = last_snapshot.get("paths", {}) or {}
    if not isinstance(paths, dict):
        return []

    observed: List[ObservedIntent] = []
    for pid, pdata in paths.items():
        if not isinstance(pdata, dict):
            continue

        src = str(pdata.get("src_host", "") or "")
        dst = str(pdata.get("dst_host", "") or "")
        protocol = infer_protocol_from_path_record(pdata, sample)
        path_nodes = normalize_path(pdata.get("path_nodes", []))
        has_traffic = path_record_has_signal(pdata)

        if not src or not dst:
            continue
        if active_only and not has_traffic:
            continue

        observed.append(
            ObservedIntent(
                src=src,
                dst=dst,
                protocol=protocol,
                path=path_nodes,
                has_traffic=has_traffic,
                e2e_delay=safe_float(pdata.get("e2e_delay_ms")),
                e2e_loss=safe_float(pdata.get("e2e_loss_rate")),
                e2e_throughput=safe_float(pdata.get("e2e_throughput_mbps")),
                raw_id=str(pid),
            )
        )

    return observed


# ============================================================
# SAFLA-style semantic comparison
# ============================================================


def compare_semantics_tuple_level(
    declared: Sequence[Dict[str, Any]],
    observed: Sequence[ObservedIntent],
) -> Dict[str, Any]:
    """
    SAFLA-vanilla style semantic comparison on (src,dst,protocol).

    missing_declared ≈ chi(i->j)=1 cases
    extra_observed   ≈ gamma(j->i)=1 cases
    """
    declared_set = {
        IntentTuple(d["src"], d["dst"], normalize_protocol(d["protocol"]))
        for d in declared
    }
    observed_set = {o.tuple_key for o in observed}

    missing_declared = declared_set - observed_set
    extra_observed = observed_set - declared_set

    return {
        "declared_set": declared_set,
        "observed_set": observed_set,
        "missing_declared": missing_declared,
        "extra_observed": extra_observed,
        "semantic_drift": int(bool(missing_declared or extra_observed)),
        "delta_s": len(missing_declared) + len(extra_observed),
    }



def compare_semantics_path_level(
    declared: Sequence[Dict[str, Any]],
    observed: Sequence[ObservedIntent],
) -> Dict[str, Any]:
    """
    Path-aware semantic comparison.

    For declared intents with a known baseline path, compare on
    (src,dst,protocol,path); otherwise fall back to tuple-level matching.
    """
    declared_path_set = set()
    declared_tuple_set = set()
    declared_without_path = set()

    for d in declared:
        tup = IntentTuple(d["src"], d["dst"], normalize_protocol(d["protocol"]))
        declared_tuple_set.add(tup)
        if d.get("declared_path"):
            declared_path_set.add(IntentPathTuple(d["src"], d["dst"], normalize_protocol(d["protocol"]), tuple(d["declared_path"])))
        else:
            declared_without_path.add(tup)

    observed_path_set = {o.path_key for o in observed}
    observed_tuple_set = {o.tuple_key for o in observed}

    # Declared intents with path must match path-aware set; otherwise fallback to tuple-level.
    missing_declared = set()
    for d in declared:
        if d.get("declared_path"):
            key = IntentPathTuple(d["src"], d["dst"], normalize_protocol(d["protocol"]), tuple(d["declared_path"]))
            if key not in observed_path_set:
                missing_declared.add(key)
        else:
            key = IntentTuple(d["src"], d["dst"], normalize_protocol(d["protocol"]))
            if key not in observed_tuple_set:
                missing_declared.add(key)

    # Extra observed = any tuple not declared, OR any path-level mismatch for declared tuples with a baseline path.
    extra_observed = set()
    declared_path_tuples = {IntentTuple(k.src, k.dst, k.protocol) for k in declared_path_set}
    declared_paths_by_tuple = {}
    for k in declared_path_set:
        declared_paths_by_tuple.setdefault(IntentTuple(k.src, k.dst, k.protocol), set()).add(k.path)

    for o in observed:
        ot = o.tuple_key
        if ot not in declared_tuple_set:
            extra_observed.add(o.path_key)
            continue
        # Tuple declared, but if this tuple has baseline-path constraints, wrong path is treated as extra semantic behavior.
        if ot in declared_path_tuples:
            allowed_paths = declared_paths_by_tuple.get(ot, set())
            if o.path not in allowed_paths:
                extra_observed.add(o.path_key)

    return {
        "missing_declared": missing_declared,
        "extra_observed": extra_observed,
        "semantic_drift": int(bool(missing_declared or extra_observed)),
        "delta_s": len(missing_declared) + len(extra_observed),
    }


# ============================================================
# Performance drift (adapted SAFLA full variant)
# ============================================================


def extract_intent_thresholds(sample: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    intent = sample.get("intent", {}) or {}
    perf = intent.get("performance_constraints", {}) or {}
    return {
        "perf": {
            "delay_threshold_ms": safe_float(perf.get("delay_threshold_ms")),
            "loss_threshold": safe_float(perf.get("loss_threshold")),
            "bandwidth_threshold_mbps": safe_float(perf.get("bandwidth_threshold_mbps")),
            "jitter_threshold_ms": safe_float(perf.get("jitter_threshold_ms")),
        }
    }



def choose_best_observation_for_declared(
    declared_intent: Dict[str, Any],
    observed: Sequence[ObservedIntent],
) -> Optional[ObservedIntent]:
    """
    Prefer an observation that matches declared tuple and declared path.
    If none exists, fall back to any tuple match with the highest throughput.
    """
    src = declared_intent["src"]
    dst = declared_intent["dst"]
    protocol = normalize_protocol(declared_intent["protocol"])
    declared_path = tuple(declared_intent.get("declared_path") or ())

    tuple_matches = [
        o for o in observed
        if o.src == src and o.dst == dst and normalize_protocol(o.protocol) == protocol
    ]
    if not tuple_matches:
        return None

    if declared_path:
        exact_path = [o for o in tuple_matches if o.path == declared_path]
        if exact_path:
            return max(exact_path, key=lambda x: (x.e2e_throughput or 0.0))

    return max(tuple_matches, key=lambda x: (x.e2e_throughput or 0.0))



def detect_performance_drift(
    declared: Sequence[Dict[str, Any]],
    observed: Sequence[ObservedIntent],
    thresholds: Dict[str, Dict[str, Optional[float]]],
) -> int:
    """
    Adapted SAFLA performance check.

    The original paper defines delta_k = p_mon - p_req and quantizes discrepancies.
    Here, because the dataset directly provides intent thresholds, we compare
    observed QoS against those thresholds as a practical approximation.
    """
    perf_th = thresholds.get("perf", {}) or {}

    for d in declared:
        best = choose_best_observation_for_declared(d, observed)
        if best is None:
            # No realized path for this declared intent: semantic drift already exists.
            # We keep perf conservative and do not double-count here.
            continue

        delay_th = perf_th.get("delay_threshold_ms")
        if delay_th is not None and best.e2e_delay is not None and best.e2e_delay > delay_th:
            return 1

        loss_th = perf_th.get("loss_threshold")
        if loss_th is not None and best.e2e_loss is not None and best.e2e_loss > loss_th:
            return 1

        bw_th = perf_th.get("bandwidth_threshold_mbps")
        if bw_th is not None and best.e2e_throughput is not None and best.e2e_throughput < bw_th:
            return 1

    return 0


# ============================================================
# Variant predictors
# ============================================================


def safla_vanilla_predict(sample: Dict[str, Any]) -> Dict[str, int]:
    declared = extract_declared_intents(sample)
    observed = extract_observed_intents(sample, active_only=True)
    sem = compare_semantics_tuple_level(declared, observed)
    return {
        "perf": 0,
        "path": int(sem["semantic_drift"]),
        "energy": 0,
    }



def safla_extended_predict(sample: Dict[str, Any]) -> Dict[str, int]:
    declared = extract_declared_intents(sample)
    observed = extract_observed_intents(sample, active_only=True)
    sem = compare_semantics_path_level(declared, observed)
    return {
        "perf": 0,
        "path": int(sem["semantic_drift"]),
        "energy": 0,
    }



def safla_full_predict(sample: Dict[str, Any]) -> Dict[str, int]:
    declared = extract_declared_intents(sample)
    observed = extract_observed_intents(sample, active_only=True)
    sem = compare_semantics_path_level(declared, observed)
    thresholds = extract_intent_thresholds(sample)
    perf = detect_performance_drift(declared, observed, thresholds)
    return {
        "perf": int(perf),
        "path": int(sem["semantic_drift"]),
        "energy": 0,
    }


# ============================================================
# Metrics
# ============================================================


def compute_metrics(all_preds: Dict[str, List[int]], all_labels: Dict[str, List[int]], total: int) -> Dict[str, Any]:
    per_clause = {}
    for name in CLAUSE_NAMES:
        p = np.asarray(all_preds[name], dtype=int)
        l = np.asarray(all_labels[name], dtype=int)
        tp = int(((p == 1) & (l == 1)).sum())
        fp = int(((p == 1) & (l == 0)).sum())
        fn = int(((p == 0) & (l == 1)).sum())
        tn = int(((p == 0) & (l == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(len(p), 1)
        per_clause[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "support_pos": int((l == 1).sum()),
            "support_neg": int((l == 0).sum()),
        }

    macro_f1 = float(np.mean([per_clause[n]["f1"] for n in CLAUSE_NAMES]))

    any_pred = np.zeros(total, dtype=int)
    any_label = np.zeros(total, dtype=int)
    for name in CLAUSE_NAMES:
        any_pred |= np.asarray(all_preds[name], dtype=int)
        any_label |= np.asarray(all_labels[name], dtype=int)

    tp = int(((any_pred == 1) & (any_label == 1)).sum())
    fp = int(((any_pred == 1) & (any_label == 0)).sum())
    fn = int(((any_pred == 0) & (any_label == 1)).sum())
    tn = int(((any_pred == 0) & (any_label == 0)).sum())
    det_precision = tp / max(tp + fp, 1)
    det_recall = tp / max(tp + fn, 1)
    det_f1 = 2 * det_precision * det_recall / max(det_precision + det_recall, 1e-8)

    return {
        "macro_f1": macro_f1,
        "per_clause": per_clause,
        "detection": {
            "precision": float(det_precision),
            "recall": float(det_recall),
            "f1": float(det_f1),
            "pos_samples": int(any_label.sum()),
            "neg_samples": int(total - any_label.sum()),
        },
    }


# ============================================================
# Evaluation
# ============================================================


def evaluate_safla(test_path: str, variant: str = "full", label_source: str = "current") -> Tuple[Dict[str, Any], Dict[str, int]]:
    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_preds = {name: [] for name in CLAUSE_NAMES}
    all_labels = {name: [] for name in CLAUSE_NAMES}

    stats = {
        "samples": len(data),
        "samples_without_declared_path": 0,
        "samples_with_extra_observed_intents": 0,
        "samples_with_missing_declared_intents": 0,
    }

    for sample in data:
        declared = extract_declared_intents(sample)
        observed = extract_observed_intents(sample, active_only=True)

        if any(not d.get("declared_path") for d in declared):
            stats["samples_without_declared_path"] += 1

        tuple_sem = compare_semantics_tuple_level(declared, observed)
        if tuple_sem["extra_observed"]:
            stats["samples_with_extra_observed_intents"] += 1
        if tuple_sem["missing_declared"]:
            stats["samples_with_missing_declared_intents"] += 1

        if variant == "vanilla":
            preds = safla_vanilla_predict(sample)
        elif variant == "extended":
            preds = safla_extended_predict(sample)
        elif variant == "full":
            preds = safla_full_predict(sample)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        gt = get_label_dict(sample, label_source=label_source)
        for name in CLAUSE_NAMES:
            all_preds[name].append(int(preds.get(name, 0)))
            all_labels[name].append(int(gt.get(name, 0)))

    metrics = compute_metrics(all_preds, all_labels, total=len(data))
    return metrics, stats


# ============================================================
# Reporting
# ============================================================


def print_results(metrics: Dict[str, Any], variant_name: str) -> None:
    print(f"\n  {variant_name}")
    print(f"  {'=' * 55}")
    print(f"\n  Detection F1:  {metrics['detection']['f1']:.4f}  "
          f"(P={metrics['detection']['precision']:.4f} "
          f"R={metrics['detection']['recall']:.4f})")
    print(f"  Macro F1:      {metrics['macro_f1']:.4f}")
    print("\n  Per-clause:")
    for name in CLAUSE_NAMES:
        c = metrics['per_clause'][name]
        note = ""
        if variant_name.startswith("SAFLA-vanilla") and name in ["perf", "energy"]:
            note = " (not detected by this variant)"
        elif variant_name.startswith("SAFLA-extended") and name in ["perf", "energy"]:
            note = " (not detected by this variant)"
        elif "SAFLA" in variant_name and name == "energy":
            note = " (not in SAFLA's scope)"
        print(f"    {name:<8}: P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1']:.3f}{note}")


# ============================================================
# Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Revised SAFLA-style baseline")
    parser.add_argument("--test", type=str, default=r'D:\datasets\Intent_drift\icsme\test.json')
    parser.add_argument("--output", type=str, default="results_safla_revised.json")
    parser.add_argument(
        "--label-source",
        type=str,
        default="current",
        choices=["current", "future"],
        help="Reactive SAFLA should normally use current labels; future is allowed for benchmarking convenience.",
    )
    args = parser.parse_args()

    print(f"Loading test data: {args.test}")
    with open(args.test, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Test samples: {len(data)}")

    results: Dict[str, Any] = {}
    extra_stats: Dict[str, Any] = {}

    variants = [
        ("vanilla", "SAFLA-vanilla [tuple-level semantic comparison]"),
        ("extended", "SAFLA-extended [path-aware semantic comparison]"),
        ("full", "SAFLA-full [path-aware semantics + QoS checks]"),
    ]

    for key, title in variants:
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
        t0 = time.time()
        metrics, stats = evaluate_safla(args.test, variant=key, label_source=args.label_source)
        elapsed = time.time() - t0
        print_results(metrics, title)
        print(f"\n  Time: {elapsed:.2f}s")
        print(f"  Extra observed intents:   {stats['samples_with_extra_observed_intents']}")
        print(f"  Missing declared intents: {stats['samples_with_missing_declared_intents']}")
        print(f"  No declared path:         {stats['samples_without_declared_path']}")
        results[key] = metrics
        extra_stats[key] = stats

    print("\n" + "=" * 70)
    print("Summary: revised SAFLA variants comparison")
    print("=" * 70)
    print(f"\n  {'Variant':<25} {'Det.F1':>8} {'Macro':>8} {'perf':>8} {'path':>8} {'energy':>8}")
    print(f"  {'-' * 70}")
    for vname, vkey in [
        ("SAFLA-vanilla", "vanilla"),
        ("SAFLA-extended", "extended"),
        ("SAFLA-full", "full"),
    ]:
        m = results[vkey]
        print(f"  {vname:<25} {m['detection']['f1']:>8.4f} {m['macro_f1']:>8.4f} "
              f"{m['per_clause']['perf']['f1']:>8.4f} "
              f"{m['per_clause']['path']['f1']:>8.4f} "
              f"{m['per_clause']['energy']['f1']:>8.4f}")

    output = {
        "method": "Revised SAFLA-style baselines (adapted from Kou et al., IEEE TCCN 2026)",
        "test_data": args.test,
        "label_source": args.label_source,
        "notes": [
            "This is an adapted baseline, not a literal reproduction of SAFLA's OpenFlow extraction pipeline.",
            "Observed intent set is approximated from all end-to-end path records in the current snapshot.",
            "Energy clause is always set to 0 because SAFLA does not model energy drift.",
        ],
        "description": {
            "vanilla": (
                "Tuple-level semantic SAFLA baseline. Compares declared intent set and observed intent set on "
                "(src,dst,protocol), capturing both missing declared intents and extra undeclared intents."
            ),
            "extended": (
                "Path-aware semantic SAFLA baseline. Adds direct path consistency checks on top of tuple-level "
                "semantic comparison."
            ),
            "full": (
                "Path-aware semantic comparison plus practical QoS threshold checks for delay/loss/throughput, "
                "approximating SAFLA's joint semantic and performance drift detection."
            ),
        },
        "stats": extra_stats,
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
