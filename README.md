
## Prerequisites

### System Requirements

- **OS**: Ubuntu 20.04.6 LTS
- **Python**: 3.10
- **RAM**: ≥ 8 GB recommended
- **Disk**: ≥ 20 GB for full dataset generation

### Software Dependencies

| Component | Version | Purpose |
|---|---|---|
| [Mininet](http://mininet.org/) | 2.3.0 | Network emulation |
| [Ryu](https://ryu-sdn.org/) | 4.34 | SDN controller (OpenFlow) |
| [Open vSwitch](https://www.openvswitch.org/) | 2.13+ | Virtual switches |
| iperf / iperf3 | 3.x | Traffic generation |
| tc (iproute2) | system default | Traffic shaping (netem) |

### Python Packages

```bash
pip install numpy pyyaml networkx scikit-learn
pip install ryu  # SDN controller
```

---

## Project Structure

```
intent_drift_platform/
├── generate_real_trace_dataset.py    # Main generation script (OCIL pipeline)
├── mininet_env/
│   └── topology.py                   # NetworkEnvironment: Mininet topology setup
├── drift_injection/
│   └── drift_injector.py             # DriftInjector: tc netem & flow table manipulation
├── data_collection/
│   ├── collector.py                  # MetricsCollector, EnergyAwareCollector
│   └── dataset_formatter.py          # DatasetFormatter, HierarchicalFeatureExtractor
├── utils/
│   ├── routing_applier.py            # RoutingApplier: OpenFlow flow rule deployment
│   ├── routing_generator.py          # RoutingGenerator: path computation
│   ├── traffic_generator.py          # TrafficGenerator: iperf-based replay
│   ├── real_traffic_replay.py        # RealTrafficReplayer, TrafficMatrixScaler
│   ├── energy_model.py               # NetworkEnergyModel: switch power estimation
│   ├── sndlib_parser.py              # SNDlibParser: GÉANT topology & demand matrix
│   ├── mawi_parser.py                # MAWIParser: packet characteristics extraction
│   └── qos_config.py                 # QoSConfigurator: HTB bandwidth limiting
├── models/
│   ├── intent.py                     # Intent, Constraint, PerformanceConstraints, ...
│   └── ...
├── data/
│   ├── sndlib/                       # GÉANT topology and demand matrices
│   └── mawi/                         # MAWI traffic traces (or synthetic profile)
├── baselines/
│   ├── baseline_rule_based.py        # Rule-based thresholding [Dzeparoska et al.]
│   ├── baseline_dbscan.py            # DBSCAN clustering [Muonagor et al.]
│   ├── baseline_lead_drift.py        # MLP risk score [Hossain & Aljoby]
│   └── baseline_random_forest.py     # Random Forest with handcrafted features
└── README.md
```

---

## Installation

### Step 1: Install Mininet

```bash
# Option A: Native install (recommended for Ubuntu)
sudo apt-get install mininet

# Option B: From source
git clone https://github.com/mininet/mininet.git
cd mininet
sudo ./util/install.sh -nfv
```

### Step 2: Install Ryu Controller

```bash
pip install ryu
```

### Step 3: Install Open vSwitch

```bash
sudo apt-get install openvswitch-switch
sudo service openvswitch-switch start
```

### Step 4: Install Python Dependencies

```bash
pip install numpy pyyaml networkx scikit-learn
```

### Step 5: Verify Installation

```bash
# Test Mininet
sudo mn --test pingall

# Test Ryu
ryu-manager --version
```

---

## Quick Start

### 1. First, open two terminals (Terminal A and Terminal B) and navigate to the project root directory.

### 2. ## Activate the virtual environment:
source ~/ryu_env_py310/bin/activate

### 3. Start Ryu Controller in Terminal A

```bash
ryu-manager ryu_controller/intent_controller.py --ofp-tcp-listen-port 6653 --wsapi-port 8080 --verbose
```

### 2. Run Data Generation in Terminal B

```bash
sudo ~/ryu_env_py310/bin/python -u experiments/generate_real_trace_dataset.py --num-intents 4  2>&1 | tee log.txt
```

### 3. Check Output

After generation completes, the output directory contains:

```
data/real_trace_dataset/
├── samples_raw_new.jsonl          # Raw snapshots (1 per line, ~1.4 GB)
├── train_new.json                 # Training sequences (70%)
├── val_new.json                   # Validation sequences (15%)
├── test_new.json                  # Test sequences (15%)
├── generation_config_new.json     # Full configuration record
└── synthetic_mawi_profile.json    # MAWI traffic profile used
```

---

## Configuration

Key parameters can be modified in `generate_real_trace_dataset.py`:

### Network & Traffic

| Parameter | Default | Description |
|---|---|---|
| Topology | GÉANT (22 switches, 58 links) | From SNDlib |
| Traffic matrix | SNDlib GÉANT demands | Scaled to 10 Mbps max |
| Packet characteristics | MAWI trace | Inter-packet gap & packet size |
| Sampling interval | 1 second | Metrics collection frequency |

The complete traffic matrix data can be found on Website [https://sndlib.put.poznan.pl/home.action] and downloaded to Directory ~/data/real_traces/sndlib/geant/.
The complete MAWI data can be found on Website [https://mawi.wide.ad.jp/mawi/] and downloaded to Directory ~/data/real_traces/mawi.

### Intent Templates

| Template | Performance margin | Energy margin | Use case |
|---|---|---|---|
| `tight_qos` | Small | Large | QoS-sensitive services |
| `balanced` | Medium | Medium | General purpose |
| `loose_energy` | Large | Small | Energy-constrained networks |

### Drift Injection

| Parameter | Default | Description |
|---|---|---|
| Events per experiment | 1-2 (weighted) | Number of drift events |
| Drift duration | 3-15s (exponential) | How long each drift lasts |
| Inter-drift gap | 15-50s (exponential) | Normal period between events |
| Performance drift delay | 10-200ms (log-normal) | Injected extra delay |
| Performance drift loss | 0.1-20% (Beta) | Injected packet loss |

### Sequence Construction

| Parameter | Default | Description |
|---|---|---|
| Window size ($T$) | 10 | Number of historical snapshots |
| Prediction horizon ($h$) | 3 | Future steps for label |
| Persistence threshold ($\pi$) | 2 | Min consecutive violations |
| Max gap | 12s | Discard windows with gaps |
| Boundary gap | $T + h$ | Leakage prevention between splits |

### OCIL Calibration

| Parameter | Default | Description |
|---|---|---|
| Upper-bound multiplier ($\alpha_u$) | 5× P99 | For delay, loss, power |
| Lower-bound multiplier ($\alpha_l$) | 0.1× P1 | For throughput, efficiency |
| Power floor | 2600 W | Absolute minimum power threshold |
| Efficiency ceiling | 0.04 | Absolute maximum efficiency threshold |

---

## Output Format

### Raw Snapshots (`samples_raw_new.jsonl`)

Each line is a JSON object:

```json
{
  "timestamp": 1776087282.0,
  "total_throughput_mbps": 45.23,
  "total_power_watts": 2542.5,
  "energy_efficiency": 0.374,
  "links": {
    "s1-s3": {"delay_ms": 0.12, "loss_rate": 0.0, "throughput_mbps": 5.2, "utilization": 0.05, "power_watts": 2.1},
    ...
  },
  "paths": {
    "h20-h18": {"src_host": "h20", "dst_host": "h18", "e2e_delay_ms": 5.3, "e2e_loss_rate": 0.0, "e2e_throughput_mbps": 2.7, "path_nodes": ["s20","s1","s5","s7","s6","s18"], "num_hops": 6},
    ...
  },
  "intent": {
    "match": {"src": "h20", "dst": "h18", "protocol": "UDP"},
    "performance_constraints": {"delay_threshold_ms": 27.5, "loss_threshold": 0.048, "bandwidth_threshold_mbps": 0.34},
    "path_constraints": {"waypoints": ["s5","s7"], "avoid_nodes": [], "max_hops": 8},
    "energy_constraints": {"max_power_watts": 2721.7, "min_efficiency_mbps_per_w": 0.002}
  },
  "clause_labels": {"perf": 0, "path": 0, "energy": 0},
  "injected_drift_type": "normal",
  "drift_location": null
}
```

### Sequence Samples (`train/val/test.json`)

Each sample is a JSON object:

```json
{
  "window": [snapshot_1, snapshot_2, ..., snapshot_T],
  "intent": { ... },
  "future_clause_labels": {"perf": 0, "path": 1, "energy": 0},
  "future_has_any_drift": true,
  "drift_location": {"links": ["s4-s17"], "type": "path"},
  "experiment_id": "geant_0_1",
  "baseline_routing_paths": { ... }
}
```

---

## Baseline Methods

Four baseline implementations are provided for benchmarking:

```bash
# Rule-based thresholding (reactive, clause-level)
python baselines/baseline_rule_based.py --test data/real_trace_dataset/test.json

# DBSCAN clustering (unsupervised, binary only)
python baselines/baseline_dbscan.py --train data/real_trace_dataset/train.json \
                                     --test data/real_trace_dataset/test.json

# LEAD-Drift MLP (proactive, binary only)
python baselines/baseline_lead_drift.py --train data/real_trace_dataset/train.json \
                                         --test data/real_trace_dataset/test.json

# Random Forest (proactive, clause-level)
python baselines/baseline_random_forest.py --train data/real_trace_dataset/train.json \
                                            --test data/real_trace_dataset/test.json
```

### Rebuilding with Different Parameters

To generate datasets with different window sizes or horizons without re-running Mininet:

```bash
# Generate all sensitivity variants from raw snapshots
python rebuild_from_raw.py --batch

# Or generate a specific variant
python rebuild_from_raw.py --window-size 5 --horizon 3 --suffix _T5
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|---|---|
| `RTNETLINK: Operation not permitted` | Run with `sudo` |
| `Error connecting to Ryu` | Ensure Ryu is running: `ryu-manager ryu.app.simple_switch_13 ryu.app.ofctl_rest --observe-links` |
| `No module named 'mininet'` | Install Mininet: `sudo apt-get install mininet` |
| `iperf: command not found` | Install iperf: `sudo apt-get install iperf iperf3` |
| All samples labeled as drift | Check OCIL calibration logs for `[threshold calibration]` messages |
| `samples_raw_new.jsonl` is empty | Check Ryu controller connection and flow table installation |

### Verifying Data Quality

```python
import json

# Check drift ratio in raw snapshots
normal = drift = 0
with open('data/real_trace_dataset/samples_raw_new.jsonl') as f:
    for line in f:
        s = json.loads(line)
        cl = s.get('clause_labels', {})
        if any(cl.get(k, 0) for k in ['perf', 'path', 'energy']):
            drift += 1
        else:
            normal += 1
print(f"Normal: {normal} ({normal/(normal+drift):.1%}), Drift: {drift} ({drift/(normal+drift):.1%})")
# Expected: Normal ~70-80%, Drift ~20-30%
```

---

## Citation

If you use this dataset or code, please cite:

```bibtex
@inproceedings{ibn-drift2026,
  title={IBN-Drift: A Reusable Benchmark Dataset for Multi-Dimensional Intent Drift in Intent-Based Network Management Systems},
  author={[Authors]},
  booktitle={Proceedings of the IEEE International Conference on Software Maintenance and Evolution (ICSME)},
  year={2026}
}
```

---

## License

This project is released under the MIT License.
