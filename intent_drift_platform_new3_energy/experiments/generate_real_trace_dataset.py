# experiments/generate_real_trace_dataset.py

import os
import sys
import time
import yaml
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mininet_env.topology import NetworkEnvironment
from drift_injection.drift_injector import (
    DriftInjector, DriftConfig, DriftType,
    create_performance_drift, create_energy_drift, create_hidden_energy_drift
)
from data_collection.collector import MetricsCollector, EnergyAwareCollector
from data_collection.dataset_formatter import DatasetFormatter, HierarchicalFeatureExtractor
from utils.traffic_generator import TrafficGenerator
from utils.routing_generator import RoutingGenerator
from utils.qos_config import QoSConfigurator

# Regarding Real Traffic
from utils.mawi_parser import MAWIParser, SyntheticMAWIProfile
from utils.sndlib_parser import SNDlibParser
from utils.real_traffic_replay import RealTrafficReplayer, TrafficMatrixScaler
from utils.routing_applier import RoutingApplier, apply_routing_to_controller
# Energy Consumption Model
from utils.energy_model import (
    NetworkEnergyModel, EnergyMetrics, EnergyDriftDetector,
    SwitchEnergyProfile, LinkEnergyProfile, create_default_energy_model
)
# Intent Parser
from utils.intent_parser import (
    Intent, IntentParser, IntentType,
    PerformanceConstraints, EnergyConstraints, PathConstraints,
    Constraint, ConstraintOperator
)


class IntentGenerator:

    # Business Type Templates
    INTENT_TEMPLATES = {
        'realtime_video': {
            'name': '实时视频流意图',
            'type': IntentType.GREEN_QOS,
            'delay_range': (20, 50),        # ms
            'loss_range': (0.001, 0.01),     # ratio
            'bandwidth_range': (10, 50),     # Mbps
            'jitter_range': (5, 15),         # ms
            'power_range': (500, 1500),      # W
            'efficiency_range': (2.0, 5.0),  # Mbps/W
            'max_hops_range': (3, 5),
        },
        'web_service': {
            'name': 'Web服务意图',
            'type': IntentType.GREEN_QOS,
            'delay_range': (50, 200),
            'loss_range': (0.01, 0.05),
            'bandwidth_range': (5, 20),
            'jitter_range': (10, 30),
            'power_range': (1000, 3000),
            'efficiency_range': (1.0, 3.0),
            'max_hops_range': (4, 8),
        },
        'bulk_transfer': {
            'name': '批量传输意图',
            'type': IntentType.GREEN_QOS,
            'delay_range': (100, 500),
            'loss_range': (0.01, 0.10),
            'bandwidth_range': (50, 200),
            'jitter_range': (20, 50),
            'power_range': (2000, 5000),
            'efficiency_range': (0.5, 2.0),
            'max_hops_range': (5, 10),
        },
        'energy_saving': {
            'name': '节能优先意图',
            'type': IntentType.ENERGY_SAVING,
            'delay_range': (100, 300),       # Relax performance requirements
            'loss_range': (0.02, 0.08),
            'bandwidth_range': (1, 10),
            'jitter_range': (20, 50),
            'power_range': (300, 800),       # Strict energy consumption
            'efficiency_range': (3.0, 8.0),
            'max_hops_range': (3, 5),
        },
    }
    
    def __init__(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
        self._intent_counter = 0
    
    def generate_intent(self, 
                        template_name: str = None,
                        src_host: str = None, 
                        dst_host: str = None,
                        path_nodes: List[str] = None) -> Intent:

        if template_name is None:
            template_name = self.rng.choice(list(self.INTENT_TEMPLATES.keys()))
        
        template = self.INTENT_TEMPLATES[template_name]
        self._intent_counter += 1
        
        # Randomly sample constraint values within the template range
        delay_threshold = float(self.rng.uniform(*template['delay_range']))
        loss_threshold = float(self.rng.uniform(*template['loss_range']))
        bandwidth_threshold = float(self.rng.uniform(*template['bandwidth_range']))
        jitter_threshold = float(self.rng.uniform(*template['jitter_range']))
        power_threshold = float(self.rng.uniform(*template['power_range']))
        efficiency_threshold = float(self.rng.uniform(*template['efficiency_range']))
        max_hops = int(self.rng.randint(*template['max_hops_range']))
        
        # Define performance constraints
        performance = PerformanceConstraints(
            delay=Constraint("end_to_end_delay", ConstraintOperator.LE, delay_threshold, "ms"),
            loss=Constraint("packet_loss_rate", ConstraintOperator.LE, loss_threshold, "ratio"),
            bandwidth=Constraint("throughput", ConstraintOperator.GE, bandwidth_threshold, "Mbps"),
            jitter=Constraint("delay_jitter", ConstraintOperator.LE, jitter_threshold, "ms"),
        )
        
        # Establishing Energy Consumption Constraints
        energy = EnergyConstraints(
            max_power=Constraint("total_network_power", ConstraintOperator.LE, power_threshold, "watts"),
            min_efficiency=Constraint("energy_efficiency", ConstraintOperator.GE, efficiency_threshold, "Mbps/W"),
        )
        
        # Define path constraints
        # Key: waypoints should not be the entire path (too rigid), but rather randomly selected 1-2 nodes from the path
        # similar to firewalls/DPI in service chains
        if path_nodes and len(path_nodes) >= 3:
            middle_nodes = path_nodes[1:-1]  # Exclude first and last
            max_wp = min(2, len(middle_nodes))
            num_waypoints = int(self.rng.randint(1, max_wp + 1))
            if num_waypoints > 0 and len(middle_nodes) > 0:
                wp_indices = self.rng.choice(
                    len(middle_nodes), size=num_waypoints, replace=False
                )
                selected_waypoints = [middle_nodes[int(i)] for i in np.atleast_1d(wp_indices)]
            else:
                selected_waypoints = []
        else:
            selected_waypoints = []
        
        path = PathConstraints(
            waypoints=selected_waypoints,
            max_hops=max_hops,
            energy_aware_routing=True,
        )
        
        # Building Traffic Matching
        match = {}
        if src_host and dst_host:
            match = {'src': src_host, 'dst': dst_host, 'protocol': 'UDP'}
        
        intent = Intent(
            intent_id=f"intent_{self._intent_counter:04d}",
            name=f"{template['name']}_{self._intent_counter}",
            intent_type=template['type'],
            match=match,
            performance=performance,
            energy=energy,
            path=path,
            priority=self.rng.randint(50, 200),
            description=f"Auto-generated {template_name} intent",
        )
        
        return intent
    
    def generate_intent_batch(self, 
                              n: int,
                              src_host: str = None,
                              dst_host: str = None,
                              path_nodes: List[str] = None) -> List[Intent]:
        """Generate a set of diverse intents (covering different business types)"""
        intents = []
        templates = list(self.INTENT_TEMPLATES.keys())
        for i in range(n):
            template = templates[i % len(templates)]
            intent = self.generate_intent(template, src_host, dst_host, path_nodes)
            intents.append(intent)
        return intents
    
    @staticmethod
    def intent_to_dict(intent: Intent) -> dict:

        result = {
            'intent_id': intent.intent_id,
            'name': intent.name,
            'type': intent.intent_type.value,
            'priority': intent.priority,
            'match': intent.match,
        }
        
        # Performance Constraints → Quantification
        if intent.performance:
            perf = intent.performance
            result['performance_constraints'] = {
                'delay_threshold_ms': perf.delay.threshold if perf.delay else None,
                'loss_threshold': perf.loss.threshold if perf.loss else None,
                'bandwidth_threshold_mbps': perf.bandwidth.threshold if perf.bandwidth else None,
                'jitter_threshold_ms': perf.jitter.threshold if perf.jitter else None,
            }
        
        # Energy Constraints → Quantification
        if intent.energy:
            eng = intent.energy
            result['energy_constraints'] = {
                'max_power_watts': eng.max_power.threshold if eng.max_power else None,
                'min_efficiency_mbps_per_w': eng.min_efficiency.threshold if eng.min_efficiency else None,
            }
        
        # Path Constraints → Quantification
        if intent.path:
            p = intent.path
            result['path_constraints'] = {
                'waypoints': p.waypoints,
                'avoid_nodes': p.avoid_nodes,
                'max_hops': p.max_hops,
            }
        
        return result
    
    @staticmethod
    def auto_label(intent: Intent, snapshot_dict: dict, 
                   path_changed: bool = False,
                   injected_drift_type: str = 'normal') -> dict:

        # ============================================================
        # 1. Construction Performance Metrics (Extract paths for intent-matched flows only)
        # ============================================================
        metrics = {
            'delay_ms': snapshot_dict.get('avg_delay_ms', 0),
            'loss_rate': snapshot_dict.get('avg_loss_rate', 0),
            'total_power_watts': snapshot_dict.get('total_power_watts', 0),
            'energy_efficiency': snapshot_dict.get('energy_efficiency', 0),
        }
        
        paths = snapshot_dict.get('paths', {})
        
        # Extract the src/dst from the intent; perform a perf check only on matching streams
        intent_match = getattr(intent, 'match', None) or {}
        if not isinstance(intent_match, dict):
            intent_match = {}
        intent_src = intent_match.get('src', '')
        intent_dst = intent_match.get('dst', '')
        
        if paths:
            # Prioritize flow paths that match the intent
            matched_path_data = None
            for pid, pdata in paths.items():
                src_host = pdata.get('src_host', '')
                dst_host = pdata.get('dst_host', '')
                if intent_src and intent_dst:
                    if ((src_host == intent_src and dst_host == intent_dst) or
                        (src_host == intent_dst and dst_host == intent_src)):
                        matched_path_data = pdata
                        break
            
            if matched_path_data:
                # Use only e2e metrics for intent-matching streams
                delay = matched_path_data.get('e2e_delay_ms', 0)
                loss = matched_path_data.get('e2e_loss_rate', 0)
                tput = matched_path_data.get('e2e_throughput_mbps', 0)
                
                if delay and delay > 0:
                    metrics['delay_ms'] = float(delay)
                if loss is not None:
                    metrics['loss_rate'] = float(loss)
                if tput and tput > 0:
                    metrics['throughput_mbps'] = float(tput)
            else:
                # No matching stream found; fallback to the network-wide average (maintaining the original logic)
                path_delays = [p.get('e2e_delay_ms', 0) for p in paths.values() if p.get('e2e_delay_ms', 0) > 0]
                path_losses = [p.get('e2e_loss_rate', 0) for p in paths.values()]
                path_throughputs = [p.get('e2e_throughput_mbps', 0) for p in paths.values() if p.get('e2e_throughput_mbps', 0) > 0]
                
                if path_delays:
                    metrics['delay_ms'] = max(metrics['delay_ms'], np.mean(path_delays))
                if path_losses:
                    metrics['loss_rate'] = max(metrics['loss_rate'], np.mean(path_losses))
                if path_throughputs:
                    metrics['throughput_mbps'] = np.mean(path_throughputs)
        
        # ============================================================
        # 2. Performance/Energy Clause Review
        # ============================================================

        effective_metrics = dict(metrics)
        has_meaningful_traffic = metrics.get('throughput_mbps', 0) > 0.01
        if not has_meaningful_traffic:
            effective_metrics['loss_rate'] = 0.0
        
        check_result = intent.check_all(effective_metrics)
        perf_ok = check_result['performance_satisfied']
        energy_ok = check_result['energy_satisfied']
        
        perf_violations = check_result.get('performance_violations', [])
        energy_violations = check_result.get('energy_violations', [])
        
        # ============================================================
        # 3. Path clause Check (semantic constraints + path availability)
        # Includes: max_hops, waypoints (mandatory), avoid_nodes (prohibited),
        #       and “path is effectively unavailable” (end-to-end delay/loss anomalies)
        # Important: Only paths in the “intent-matching stream” are checked, not all paths across the entire network.
        # ============================================================
        path_ok = True
        path_violations = []
        derived_drift_location = []  # List of suspicious links derived from observed anomalous paths
        
        if intent.path:
            waypoints = intent.path.waypoints if intent.path.waypoints else []
            avoid_nodes = intent.path.avoid_nodes if intent.path.avoid_nodes else []
            max_hops = intent.path.max_hops
            
            # Extract the stream that matches the intent
            intent_match = getattr(intent, 'match', None) or {}
            if not isinstance(intent_match, dict):
                intent_match = {}
            intent_src = intent_match.get('src', '')
            intent_dst = intent_match.get('dst', '')
            
            # Extract the delay threshold for intent detection, used for “path unavailable” detection
            intent_delay_threshold = None
            if intent.performance and intent.performance.delay:
                intent_delay_threshold = intent.performance.delay.threshold
            
            for path_id, p_data in paths.items():
                src_host = p_data.get('src_host', '')
                dst_host = p_data.get('dst_host', '')
                
                # Only check paths that match the intent
                # If the intent doesn't specify src/dst, fall back to checking all paths
                if intent_src and intent_dst:
                    path_matches_intent = (
                        (src_host == intent_src and dst_host == intent_dst) or
                        (src_host == intent_dst and dst_host == intent_src)
                    )
                    if not path_matches_intent:
                        continue
                
                path_nodes = p_data.get('path_nodes', [])
                num_hops = p_data.get('num_hops', 0)
                
                # 3.1 Hop count constraint
                if max_hops > 0 and num_hops > max_hops:
                    path_ok = False
                    path_violations.append(
                        f"hops[{path_id}]: {num_hops} > {max_hops}"
                    )
                
                # 3.2 Waypoint Constraints (Mandatory Waypoints)
                if waypoints and path_nodes:
                    missing = [wp for wp in waypoints if wp not in path_nodes]
                    if missing:
                        path_ok = False
                        path_violations.append(
                            f"waypoints[{path_id}]: missing {missing}"
                        )
                
                # 3.3 avoid_nodes constraint (prohibited nodes)
                if avoid_nodes and path_nodes:
                    forbidden = [an for an in avoid_nodes if an in path_nodes]
                    if forbidden:
                        path_ok = False
                        path_violations.append(
                            f"avoid_nodes[{path_id}]: forbidden {forbidden}"
                        )
                
        
        # Note: The `path` tag is no longer overridden based on the injection type.
        # The path clause must be derived entirely from the path semantics observed in the snapshot,
        # Avoid mixing labels into the generator's prior.
        
        # ============================================================
        # 4. Clause-level Clause-level multi-label output
        # ============================================================
        clause_labels = {
            'perf': int(not perf_ok),
            'path': int(not path_ok),
            'energy': int(not energy_ok),
        }
        has_any_drift = any(clause_labels.values())
        
        # ============================================================
        # 5. Backward compatibility: Derive a single `drift_label` (in order of priority: path > perf > energy)
        # ============================================================
        if not has_any_drift:
            drift_label, drift_type = 0, 'normal'
        elif clause_labels['path']:
            drift_label, drift_type = 2, 'path'
        elif clause_labels['perf']:
            drift_label, drift_type = 1, 'performance'
        elif clause_labels['energy']:
            drift_label, drift_type = 3, 'energy'
        else:
            drift_label, drift_type = 0, 'normal'
        
        all_violations = perf_violations + energy_violations + path_violations
        
        return {
            'clause_labels': clause_labels,
            'has_any_drift': has_any_drift,
            'drift_label': drift_label,
            'drift_type': drift_type,
            'performance_satisfied': perf_ok,
            'energy_satisfied': energy_ok,
            'path_satisfied': path_ok,
            'violations': all_violations,
            'path_violations': path_violations,
            'derived_drift_location': derived_drift_location,
            'auto_labeled': True,
            'metrics_used': metrics,
        }


class RealTraceDatasetGenerator:
    
    # Configuration Constants (see BNN-UPC)
    TOPOLOGIES = {
        'train': ['geant'],
        'test': ['abilene', 'germany50', 'nobel-germany']
    }
    
    NUM_ROUTING_VARIANTS = 26  # 26 routing variants for each topology
    
    SCHEDULING_POLICIES = ['FIFO', 'SP', 'WFQ', 'DRR']
    NUM_SCHEDULING_CONFIGS = 100
    
    QUEUE_SIZES = [8000, 16000, 32000, 64000]  # bits
    
    # Drift Type Configuration
    DRIFT_TYPES = {
        'normal': 0,
        'performance': 1,
        'path': 2,
        'energy': 3  # New: Energy Consumption Drift
    }
    
    def __init__(self, config):
        self.config = config
        self.output_dir = config.get('output_dir', 'data/real_trace_dataset')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize the parser
        self.sndlib_parser = SNDlibParser(config.get('sndlib_dir', 'data/real_traces/sndlib'))
        self.traffic_profile = None
        
        # Initialize the energy consumption model
        self.energy_model = create_default_energy_model()
        
        # Configure energy consumption parameters
        energy_config = config.get('energy', {})
        self.energy_model.switch_profile = SwitchEnergyProfile(
            P_chassis=energy_config.get('switch_chassis_power', 100.0),
            P_idle_per_port=energy_config.get('switch_idle_port_power', 2.0),
            P_active_per_port=energy_config.get('switch_active_port_power', 5.0),
            E_per_gbps=energy_config.get('switch_dynamic_power', 10.0)
        )
        self.energy_model.link_profile = LinkEnergyProfile(
            P_transceiver=energy_config.get('link_transceiver_power', 1.0),
            E_per_gbps=energy_config.get('link_dynamic_power', 0.5)
        )
        
        self.intent_generator = IntentGenerator(
            seed=config.get('intent_seed', 42)
        )
        # How many different intents are generated for annotation in each experiment
        self.num_intents_per_experiment = config.get('num_intents_per_experiment', 3)
        
        # Keep the existing `intent_max_power` as the fallback default value
        self.intent_max_power = config.get('intent_max_power', 1500.0)
        
        # Drift Distribution Configuration
        self.drift_distribution = config.get('drift_distribution', {
            'normal': 0.4,
            'performance': 0.2,
            'path': 0.2,
            'energy': 0.2
        })
        # Router
        self.routing_applier = RoutingApplier(
            config.get('controller_url', 'http://127.0.0.1:8080')
        )

    def prepare_traffic_profile(self):
        """Prepare the traffic profile"""
        profile_path = self.config.get('traffic_profile_path')
    
        # Check if there is a real statistical feature file
        mawi_stats_path = self.config.get('mawi_stats_path', 'profiles/mawi_real_stats.json')
    
        if profile_path and os.path.exists(profile_path):
            # Use a real MAWI PCAP (not recommended; it will be very slow)
            print("Parsing real MAWI pcap file...")
            parser = MAWIParser(profile_path)
            parser.parse_pcap(max_packets=self.config.get('max_packets', 50000))
            self.traffic_profile = parser.generate_traffic_profile()
        else:
            # Generate synthetic traffic using statistical characteristics (Recommended)
            print("Generating synthetic MAWI profile...")
        
            if os.path.exists(mawi_stats_path):
                # Load real statistical features
                synth = SyntheticMAWIProfile(profile_path=mawi_stats_path)
                print(f"  Using real statistics from {mawi_stats_path}")
            else:
                # Use default hardcoded features
                synth = SyntheticMAWIProfile()
                print("  Using default (hardcoded) statistics")
        
            profile_output = os.path.join(self.output_dir, 'synthetic_mawi_profile.json')
            self.traffic_profile = synth.save_profile(
                profile_output,
                num_packets=10000,
                scale_factor=self.config.get('time_scale_factor', 1.0)
            )
        return self.traffic_profile
    
    def prepare_topologies(self):
        """Prepare the topology configurations"""
        topologies = {}
        
        # Only use the topologies specified in the configuration
        topo_list = self.config.get('topologies', self.TOPOLOGIES['train'])
        if isinstance(topo_list, str):
            topo_list = [topo_list]
        
        for topo_name in topo_list:
            print(f"Preparing topology: {topo_name}")
            
            # Try to load from SNDlib
            network_data = self.sndlib_parser.parse_network(topo_name)
            
            if network_data:
                # Convert to Mininet configuration
                topo_config, node_map = self.sndlib_parser.convert_to_mininet_topology(network_data)
                traffic_matrix = self.sndlib_parser.get_random_traffic_matrix(network_data)
                
                topologies[topo_name] = {
                    'config': topo_config,
                    'node_map': node_map,
                    'traffic_matrix': traffic_matrix,
                    'source': 'sndlib',
                    'num_nodes': len(network_data.get('nodes', [])),
                    'num_links': len(network_data.get('links', []))
                }
                
                # Initializing the topology of the energy consumption model
                self._init_energy_model_topology(topo_config)
                
            else:
                # Use a predefined configuration
                config_path = f"configs/topologies/{topo_name}.yaml"
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        topo_config = yaml.safe_load(f)
                    topologies[topo_name] = {
                        'config': topo_config,
                        'traffic_matrix': None,
                        'source': 'predefined'
                    }
                    self._init_energy_model_topology(topo_config)
                else:
                    print(f"Warning: Topology {topo_name} not found")
        
        return topologies
    
    def _init_energy_model_topology(self, topo_config):
        """Initialize the topological information of the energy consumption model"""
        switches = []
        links = []
        
        # Extract switches and links from the topology configuration
        if 'switches' in topo_config:
            for sw in topo_config['switches']:
                switches.append({
                    'id': sw.get('name', sw.get('id')),
                    'num_ports': sw.get('num_ports', 24)
                })
        
        if 'links' in topo_config:
            for link in topo_config['links']:
                src = link.get('src', link.get('source'))
                dst = link.get('dst', link.get('target'))
                links.append({
                    'id': f"{src}-{dst}",
                    'src': src,
                    'dst': dst,
                    'capacity': link.get('bandwidth', 100)
                })
        
        if switches and links:
            self.energy_model.set_topology(switches, links)
    
    def generate_scheduling_configs(self, num_configs=100):
        """Generate scheduling policy configurations"""
        configs = []
        
        for i in range(num_configs):
            config = {
                'id': i,
                'nodes': {}
            }
            
            # Random selection strategy for each node
            for node_idx in range(50):  # Assuming a maximum of 50 nodes
                policy = np.random.choice(self.SCHEDULING_POLICIES)
                queue_size = np.random.choice(self.QUEUE_SIZES)
                
                node_config = {
                    'policy': policy,
                    'queue_size': queue_size
                }
                
                # If it is WFQ or DRR, add a weight
                if policy in ['WFQ', 'DRR']:
                    weights = np.random.dirichlet([1, 1, 1]) * 100
                    node_config['weights'] = {
                        'tos0': float(weights[0]),
                        'tos1': float(weights[1]),
                        'tos2': float(weights[2])
                    }
                
                config['nodes'][f's{node_idx + 1}'] = node_config
            
            configs.append(config)
        
        return configs
    
    # ============================================================
    # Baseline Adaptive: Adjusts the intention threshold based on the actual simulation state
    # 
    # Background: Hard-coded thresholds in intent templates (e.g., efficiency ≥ 0.5 Mbps/W)
    # far exceeding the actual performance observed in Mininet simulations (actual efficiency ≈ 0.006 Mbps/W),
    # This causes all snapshots in the “normal” segment to be incorrectly flagged as energy clause violations.
    # 
    # Solution: At the start of each experiment, first capture a baseline normal snapshot,
    # Dynamically generate intent thresholds using the median of the baseline metrics, ensuring that the “normal” segment is highly likely to satisfy the constraints.
    # ============================================================
    def _collect_baseline_metrics(self, collector, duration: int = 10) -> Dict[str, float]:

        print(f"    [baseline] Collecting baseline metrics ({duration}s)...")
        collector.start_collection(interval=1.0)
        time.sleep(duration)
        baseline_snaps = collector.stop_collection()
        
        if not baseline_snaps:
            print("      Warning: No baseline snapshots collected, using defaults")
            return {
                'delay_ms': 5.0,
                'loss_rate': 0.0,
                'throughput_mbps': 1.0,
                'total_power': 500.0,
                'efficiency': 0.006,
            }
        
        # Collect all path-level metrics (each stream at each time step is a data point)
        all_path_delays = []
        all_path_losses = []
        all_path_throughputs = []
        powers = []
        effs = []
        
        for snap in baseline_snaps:
            d = snap.to_dict() if hasattr(snap, 'to_dict') else snap
            
            paths = d.get('paths', {})
            if paths:
                for p in paths.values():
                    delay = p.get('e2e_delay_ms', 0)
                    if delay > 0:
                        all_path_delays.append(float(delay))
                    
                    loss = p.get('e2e_loss_rate', 0)
                    if loss >= 0:
                        all_path_losses.append(float(loss))
                    
                    thr = p.get('e2e_throughput_mbps', 0)
                    if thr > 0:
                        all_path_throughputs.append(float(thr))
            
            # “power” and “efficiency” are system-wide metrics; retain the global settings
            if d.get('total_power_watts', 0) > 0:
                powers.append(float(d.get('total_power_watts', 0)))
            if d.get('energy_efficiency', 0) > 0:
                effs.append(float(d.get('energy_efficiency', 0)))
        
        # Key: Set the throughput to the 30th percentile (so that 70% of the traffic can achieve this level)
        # The median is too strict (half of the traffic doesn't meet it), so we're using the P30.
        def percentile_or_default(values, p, default):
            if not values:
                return default
            return float(np.percentile(values, p))
        
        baseline = {
            'delay_ms': percentile_or_default(all_path_delays, 50, 5.0),
            'loss_rate': percentile_or_default(all_path_losses, 50, 0.0),
            'throughput_mbps': percentile_or_default(all_path_throughputs, 30, 1.0),
            'total_power': float(np.median(powers)) if powers else 500.0,
            'efficiency': float(np.median(effs)) if effs else 0.006,
        }
        
        # Print additional distribution information to aid in troubleshooting
        if all_path_throughputs:
            print(f"      path-level throughput stats: "
                  f"n={len(all_path_throughputs)}, "
                  f"min={min(all_path_throughputs):.3f}, "
                  f"P30={baseline['throughput_mbps']:.3f}, "
                  f"median={np.median(all_path_throughputs):.3f}, "
                  f"max={max(all_path_throughputs):.3f}Mbps")
        
        print(f"      baseline: delay={baseline['delay_ms']:.2f}ms, "
              f"loss={baseline['loss_rate']:.4f}, "
              f"throughput={baseline['throughput_mbps']:.3f}Mbps (P30 of single-flow), "
              f"power={baseline['total_power']:.0f}W, "
              f"efficiency={baseline['efficiency']:.4f}Mbps/W")
        return baseline
    
    def _compute_adaptive_intent_templates(self, baseline: Dict[str, float]) -> Dict[str, dict]:

        b = baseline
        
        # Absolute value protection for lower/upper bounds (calibrated with real-world data)
        POWER_FLOOR = 2600.0      # Normal operation max=2545W, add ~55W margin
        EFFICIENCY_CEIL = 0.04    # Normal operation min≈0.05, leave margin
        
        # tight_qos: Strict performance requirements, energy-efficient
        tight_qos = {
            'name': '严格QoS意图',
            'type': IntentType.GREEN_QOS,
            'delay_range':      (max(b['delay_ms'] * 4, 10),  max(b['delay_ms'] * 8, 30)),
            'loss_range':       (max(b['loss_rate'] + 0.01, 0.02),  max(b['loss_rate'] + 0.03, 0.05)),
            'bandwidth_range':  (b['throughput_mbps'] * 0.2,  b['throughput_mbps'] * 0.4),
            'jitter_range':     (max(b['delay_ms'] * 4, 10),  max(b['delay_ms'] * 8, 25)),
            'power_range':      (max(b['total_power'] * 1.5, POWER_FLOOR),
                                 max(b['total_power'] * 2.5, POWER_FLOOR * 1.3)),
            'efficiency_range': (min(b['efficiency'] * 0.3, EFFICIENCY_CEIL),
                                 min(b['efficiency'] * 0.5, EFFICIENCY_CEIL)),
            'max_hops_range':   (4, 7),
        }
        
        # balanced: Balanced Intent
        balanced = {
            'name': '平衡型意图',
            'type': IntentType.GREEN_QOS,
            'delay_range':      (max(b['delay_ms'] * 6, 20),  max(b['delay_ms'] * 10, 50)),
            'loss_range':       (max(b['loss_rate'] + 0.02, 0.03),  max(b['loss_rate'] + 0.05, 0.08)),
            'bandwidth_range':  (b['throughput_mbps'] * 0.15,  b['throughput_mbps'] * 0.3),
            'jitter_range':     (max(b['delay_ms'] * 6, 15),  max(b['delay_ms'] * 12, 40)),
            'power_range':      (max(b['total_power'] * 1.4, POWER_FLOOR),
                                 max(b['total_power'] * 2.2, POWER_FLOOR * 1.35)),
            'efficiency_range': (min(b['efficiency'] * 0.4, EFFICIENCY_CEIL),
                                 min(b['efficiency'] * 0.6, EFFICIENCY_CEIL)),
            'max_hops_range':   (5, 8),
        }
        
        # loose_energy: Flexible performance, strict energy consumption (Core message of the paper: prioritizing energy efficiency)
        loose_energy = {
            'name': '节能优先意图',
            'type': IntentType.ENERGY_SAVING,
            'delay_range':      (max(b['delay_ms'] * 8, 30),  max(b['delay_ms'] * 12, 80)),
            'loss_range':       (max(b['loss_rate'] + 0.03, 0.05),  max(b['loss_rate'] + 0.08, 0.12)),
            'bandwidth_range':  (b['throughput_mbps'] * 0.1,  b['throughput_mbps'] * 0.2),
            'jitter_range':     (max(b['delay_ms'] * 8, 20),  max(b['delay_ms'] * 15, 60)),
            # The power threshold for energy-saving mode is slightly tight but still higher than normal operating conditions.
            'power_range':      (max(b['total_power'] * 1.2, POWER_FLOOR),
                                 max(b['total_power'] * 1.8, POWER_FLOOR * 1.15)),
            'efficiency_range': (min(b['efficiency'] * 0.5, EFFICIENCY_CEIL),
                                 min(b['efficiency'] * 0.7, EFFICIENCY_CEIL * 1.2)),
            'max_hops_range':   (4, 7),
        }
        
        return {
            'tight_qos': tight_qos,
            'balanced': balanced,
            'loose_energy': loose_energy,
        }
    

    def _label_snapshots(self, snapshots, intent, injected_drift_type='normal',
                         drift_location=None, drift_params=None,
                         topo_name='', sched_config_id=0,
                         baseline_routing=None):

        samples = []
        intent_dict = IntentGenerator.intent_to_dict(intent)
        
        # Convert the tuple keys in `baseline_routing` to strings (JSON does not support tuple keys)
        # Format: “src-dst” -> [s1, s5, ...]
        baseline_routing_str = {}
        if baseline_routing and 'paths' in baseline_routing:
            for key, path in baseline_routing['paths'].items():
                if isinstance(key, tuple) and len(key) == 2:
                    str_key = f"{key[0]}-{key[1]}"
                else:
                    str_key = str(key)
                baseline_routing_str[str_key] = list(path) if path else []
        
        for snapshot in snapshots:
            sample = snapshot.to_dict()
            
            # ============================================================
            # Key point: Automatic annotation (rather than hard-coding)
            # ============================================================
            label_result = IntentGenerator.auto_label(
                intent=intent,
                snapshot_dict=sample,
                path_changed=(injected_drift_type == 'path'),
                injected_drift_type=injected_drift_type,
            )
            
            # ============================================================
            # Write clause-level multi-tags (core fields)
            # ============================================================
            sample['clause_labels'] = label_result['clause_labels']
            sample['has_any_drift'] = label_result['has_any_drift']
            
            # Backward compatibility: single drift_label
            sample['drift_label'] = label_result['drift_label']
            sample['drift_type'] = label_result['drift_type']
            sample['label'] = label_result['drift_label']
            
            # Write intent (complete constraint information, for Intent Encoder use)
            sample['intent'] = intent_dict
            
            # Write detailed auto-labeling information
            sample['label_info'] = {
                'auto_labeled': True,
                'performance_satisfied': label_result['performance_satisfied'],
                'energy_satisfied': label_result['energy_satisfied'],
                'path_satisfied': label_result['path_satisfied'],
                'violations': label_result['violations'],
                'path_violations': label_result.get('path_violations', []),
                'metrics_used': label_result['metrics_used'],
                'injected_drift_type': injected_drift_type,
            }
            
            # Write drift injection information (ground truth)
            # Prioritize using the drift_location at injection time (the actual injection location);
            # If no injection occurred (occasional violations in the normal segment), fall back to using auto_label to derive the location from the observed path
            if drift_location:
                sample['drift_location'] = drift_location
            elif label_result.get('has_any_drift') and label_result.get('derived_drift_location'):
                # During a normal segment, a clause violation is detected (such as a sudden spike in path delay),
                # “Links on abnormal paths” extracted from `auto_label` are used as potential locations
                sample['drift_location'] = list(label_result['derived_drift_location'])
            
            if drift_params:
                sample['drift_params'] = drift_params
            
            # Metadata
            sample['topology'] = topo_name
            sample['scheduling_config'] = sched_config_id
            
            # Required fields for the SAFLA-style baseline:
            # Record the initial routing (declared intent path set) at the start of the experiment,
            # For future comparison: I vs. Î
            if baseline_routing_str:
                sample['baseline_routing_paths'] = baseline_routing_str
            
            # Synchronize the label of the snapshot object
            snapshot.label = label_result['drift_label']
            snapshot.drift_type = label_result['drift_type']
            
            samples.append(sample)
        
        return samples
    
    def _sample_uniform_int(self, low: int, high: int) -> int:
        """Sample an integer duration/value in [low, high]."""
        return int(np.random.randint(low, high + 1))

    def _sample_clipped_exponential(self, scale: float, min_value: float, max_value: float, offset: float = 0.0) -> float:
        """Sample from an exponential distribution with clipping."""
        value = float(np.random.exponential(scale=scale) + offset)
        return float(np.clip(value, min_value, max_value))

    def _sample_drift_duration(self) -> int:
        cfg = self.config.get('stochastic_drift', {})
        return int(round(self._sample_clipped_exponential(
            scale=cfg.get('drift_duration_scale', 8.0),
            min_value=cfg.get('drift_duration_min', 3.0),
            max_value=cfg.get('drift_duration_max', 30.0),
            offset=cfg.get('drift_duration_offset', 2.0),
        )))

    def _sample_inter_drift_gap(self) -> int:
        cfg = self.config.get('stochastic_drift', {})
        return int(round(self._sample_clipped_exponential(
            scale=cfg.get('inter_drift_gap_scale', 8.0),
            min_value=cfg.get('inter_drift_gap_min', 2.0),
            max_value=cfg.get('inter_drift_gap_max', 25.0),
        )))

    def _sample_num_episodes(self) -> int:
        cfg = self.config.get('stochastic_drift', {})
        choices = cfg.get('episodes_choices', [1, 2, 3, 4])
        probs = cfg.get('episodes_probs', [0.3, 0.4, 0.2, 0.1])
        probs = np.array(probs, dtype=float)
        probs = probs / probs.sum()
        return int(np.random.choice(choices, p=probs))

    def _sample_target_links(self, links, k=1) -> List:
        """Uniformly sample target links without assuming any internal link schema."""
        if not links:
            return []
        k = min(int(k), len(links))
        idxs = np.random.choice(len(links), size=k, replace=False)
        return [links[int(i)] for i in np.atleast_1d(idxs)]

    def _sample_performance_drift_params(self) -> Dict[str, float]:
        cfg = self.config.get('stochastic_drift', {})
        delay_ms = int(np.clip(
            np.random.lognormal(
                mean=cfg.get('delay_lognormal_mean', 3.5),
                sigma=cfg.get('delay_lognormal_sigma', 0.8),
            ),
            cfg.get('delay_min', 10),
            cfg.get('delay_max', 200),
        ))
        loss_rate = float(np.clip(
            np.random.beta(
                cfg.get('loss_beta_a', 2.0),
                cfg.get('loss_beta_b', 20.0),
            ),
            cfg.get('loss_min', 0.001),
            cfg.get('loss_max', 0.20),
        ))
        return {'delay_ms': delay_ms, 'loss_rate': loss_rate}

    def _sample_secondary_energy_targets(self, links, topo_config, routing) -> List:
        cfg = self.config.get('stochastic_drift', {})
        secondary_size = int(cfg.get('secondary_energy_target_size', 2))
        return self._sample_target_links(links, k=secondary_size)

    def _normalize_event_type_probs(self) -> Tuple[List[str], np.ndarray]:
        dist_cfg = self.config.get('drift_distribution', {})
        event_types = ['performance', 'path', 'energy']
        probs = np.array([float(dist_cfg.get(t, 0.0)) for t in event_types], dtype=float)
        if probs.sum() <= 0:
            probs = np.array([1.0, 1.0, 1.0], dtype=float)
        probs = probs / probs.sum()
        return event_types, probs

    def _sample_event_type(self) -> str:
        event_types, probs = self._normalize_event_type_probs()
        return str(np.random.choice(event_types, p=probs))

    def _sample_secondary_type(self, primary_type: str) -> Optional[str]:
        event_types, probs = self._normalize_event_type_probs()
        filtered = [(t, p) for t, p in zip(event_types, probs) if t != primary_type]
        if not filtered:
            return None
        types = [t for t, _ in filtered]
        p = np.array([pp for _, pp in filtered], dtype=float)
        p = p / p.sum()
        return str(np.random.choice(types, p=p))

    def _build_event_schedule(self) -> Tuple[int, int, List[dict]]:
        cfg = self.config.get('stochastic_drift', {})
        normal_pre_dur = self._sample_uniform_int(
            cfg.get('normal_pre_min', 20),
            cfg.get('normal_pre_max', 45),
        )
        normal_post_dur = self._sample_uniform_int(
            cfg.get('normal_post_min', 15),
            cfg.get('normal_post_max', 40),
        )
        num_events = self._sample_num_episodes()

        schedule = []
        total_runtime = normal_pre_dur + normal_post_dur
        for idx in range(num_events):
            gap_before = self._sample_inter_drift_gap()
            drift_duration = self._sample_drift_duration()
            settle_time = self._sample_uniform_int(
                cfg.get('settle_time_min', 1),
                cfg.get('settle_time_max', 3),
            )
            primary = self._sample_event_type()
            event_types = [primary]
            if np.random.random() < float(cfg.get('mixed_drift_probability', 0.10)):
                secondary = self._sample_secondary_type(primary)
                if secondary and secondary not in event_types:
                    event_types.append(secondary)

            event = {
                'event_id': idx,
                'gap_before': gap_before,
                'settle_time': settle_time,
                'duration': drift_duration,
                'types': event_types,
            }
            schedule.append(event)
            total_runtime += gap_before + settle_time + drift_duration

        total_runtime += int(cfg.get('replay_tail_buffer', 10))
        return normal_pre_dur, normal_post_dur, schedule

    def _inject_event_drifts(self, event: dict, drift_injector, links, topo_data, routing,
                             intents=None):

        drift_params = {
            'event_id': int(event.get('event_id', -1)),
            'event_types': list(event.get('types', [])),
            'duration': int(event.get('duration', 0)),
            'settle_time': int(event.get('settle_time', 0)),
            'gap_before': int(event.get('gap_before', 0)),
        }
        drift_locations = []
        extra_energy_detection = False
        
        # Extract the (src, dst) from intents to create an intent stream for realistic path drift rerouting
        intent_flow_keys = []
        if intents:
            for intent in intents:
                m = getattr(intent, 'match', None) or {}
                if isinstance(m, dict):
                    s, d = m.get('src'), m.get('dst')
                    if s and d:
                        intent_flow_keys.append((s, d))

        for drift_type in event.get('types', []):
            if not links:
                continue
            if drift_type == 'performance':
                target_links = self._sample_target_links(links, k=1)
                if not target_links:
                    continue
                perf_params = self._sample_performance_drift_params()
                perf_drift = create_performance_drift(
                    target_links,
                    delay_ms=perf_params['delay_ms'],
                    loss_rate=perf_params['loss_rate']
                )
                drift_injector.inject_drift(perf_drift)
                drift_locations.extend(target_links)
                drift_params.setdefault('performance', []).append({
                    'targets': list(target_links),
                    **perf_params,
                })
            elif drift_type == 'path':
                # Option A: True rerouting
                # No longer passes `target_links` (previously used to add a 1000ms delay to the link),
                # Change it to pass `affected_intent_flows`, so that `drift_injector` can actively modify the flow table
                if not intent_flow_keys:
                    print("  [_inject_event_drifts] WARNING: path drift requested "
                          "but no intent flows available, skipping")
                    continue
                
                path_drift = DriftConfig(
                    drift_type=DriftType.PATH,
                    affected_intent_flows=list(intent_flow_keys),
                )
                ok = drift_injector.inject_drift(path_drift)
                
                # Collect “affected links” for drift_location annotation
                # Read the old and new paths from `path_drift_backup`, and treat the changed edges as `drift_location`
                if ok and drift_injector.path_drift_backup:
                    affected_link_changes = []
                    for flow, old_path in drift_injector.path_drift_backup.items():
                        # Find the corresponding new path
                        for d in drift_injector.active_drifts:
                            if d.get('mode') == 'reroute' and d.get('flow') == flow:
                                new_path = d.get('new_path', [])
                                # Extract links from the new path
                                for i in range(len(new_path) - 1):
                                    affected_link_changes.append(f"{new_path[i]}-{new_path[i+1]}")
                                break
                    drift_locations.extend(affected_link_changes)
                
                drift_params.setdefault('path', []).append({
                    'affected_flows': [f"{f[0]}-{f[1]}" for f in intent_flow_keys],
                    'mode': 'reroute',
                })
            elif drift_type == 'energy':
                target_links = self._sample_target_links(links, k=min(3, len(links)))
                if not target_links:
                    continue
                energy_drift = create_hidden_energy_drift(target_links)
                drift_injector.inject_drift(energy_drift)
                drift_locations.extend(target_links)
                drift_params.setdefault('energy', []).append({
                    'targets': list(target_links),
                })
                extra_energy_detection = True

        drift_location = None
        if drift_locations:
            # Remove duplicates while preserving the original order as much as possible
            seen = []
            for item in drift_locations:
                if item not in seen:
                    seen.append(item)
            drift_location = seen

        injected_type = '+'.join(event.get('types', [])) if event.get('types') else 'normal'
        return injected_type, drift_location, drift_params, extra_energy_detection

    def _collect_phase_samples(self, collector, intents, duration, topo_name, sched_config_id,
                               injected_drift_type='normal', drift_location=None,
                               drift_params=None, extra_energy_detection=False,
                               baseline_routing=None):
        collector.start_collection(interval=1.0)
        time.sleep(max(1, int(round(duration))))
        snapshots = collector.stop_collection()

        phase_samples = []
        for intent in intents:
            labeled = self._label_snapshots(
                snapshots, intent,
                injected_drift_type=injected_drift_type,
                drift_location=drift_location,
                drift_params=drift_params,
                topo_name=topo_name,
                sched_config_id=sched_config_id,
                baseline_routing=baseline_routing,
            )
            if extra_energy_detection:
                for s in labeled:
                    snapshot_for_detect = [snap for snap in snapshots if snap.timestamp == s.get('timestamp')]
                    if snapshot_for_detect:
                        drift_result = collector.detect_energy_drift(
                            snapshot_for_detect[0],
                            intent_max_power=intent.energy.max_power.threshold if intent.energy and intent.energy.max_power else self.intent_max_power,
                            performance_satisfied=s.get('label_info', {}).get('performance_satisfied', True)
                        )
                        s['energy_drift_detected'] = drift_result.get('has_drift', False)
                        s['energy_drift_severity'] = drift_result.get('severity', 0.0)
            phase_samples.extend(labeled)
        return phase_samples, snapshots

    def run_single_experiment(self, topo_name, topo_data, routing, sched_config, traffic_matrix):

        # Save the temporary topology configuration
        temp_topo_path = os.path.join(self.output_dir, 'temp_topology.yaml')
        with open(temp_topo_path, 'w') as f:
            yaml.dump(topo_data['config'], f)
        
        # Set up a network environment
        network_env = NetworkEnvironment(
            topo_config=temp_topo_path,
            controller_ip=self.config.get('controller_ip', '127.0.0.1'),
            controller_port=self.config.get('controller_port', 6653)
        )
        samples = []
        routing_verification = None
        try:
            net = network_env.start()
            time.sleep(3)
            # Disabling IPv6
            print("Disabling IPv6 on all hosts...")
            for host in net.hosts:
                host.cmd('sysctl -w net.ipv6.conf.all.disable_ipv6=1')
                host.cmd('sysctl -w net.ipv6.conf.default.disable_ipv6=1')
            
            print("Waiting for Ryu Controller to sync topology...")
            time.sleep(10)

            # ============================================================
            # Applying routing configuration
            # ============================================================
            print(f"Applying routing configuration: {routing.get('type', 'unknown')}...")
            if not self.routing_applier.apply_routing_from_mininet(routing, net):
                print("Warning: Failed to apply routing via API, will use default forwarding")
            time.sleep(2)
            
            # Verifying routing configuration
            print("Verifying routing configuration...")
            routing_verification = self.routing_applier.verify_routing()
            if routing_verification.get('is_valid'):
                print(f"  ✓ Routing verified: {len(routing_verification.get('configured_routes', {}))} routes")
                print(f"  ✓ Installed flows: {len(routing_verification.get('installed_flows', []))}")
            else:
                issues = routing_verification.get('issues', [])
                print(f"  ✗ Routing verification failed: {len(issues)} issues")
                for issue in issues[:5]:
                    print(f"    - {issue}")

            # Warming up network
            #print("Warming up network...")
            #for i in range(3):
            #    loss = net.pingAll()
            #    print(f"  Warm-up round {i+1}: {loss}% loss")
            #    if loss == 0:
            #        break
            #    time.sleep(2)
            
            # Configure QoS
            qos_config = QoSConfigurator(network_env)
            self._apply_scheduling_config(qos_config, sched_config)
            time.sleep(1)
            
            # Initialize traffic replay
            replayer = RealTrafficReplayer(network_env)
            if self.traffic_profile:
                replayer.traffic_profile = self.traffic_profile
            
            # Scale the Traffic Matrix
            if traffic_matrix is not None:
                scaled_tm = TrafficMatrixScaler.scale_matrix(
                    traffic_matrix,
                    target_max_rate=self.config.get('max_traffic_rate', 10.0)
                )
                scaled_tm = TrafficMatrixScaler.add_random_variation(scaled_tm, 0.1)
            else:
                n = len(net.hosts)
                scaled_tm = np.random.uniform(0, 5000, (n, n))
                np.fill_diagonal(scaled_tm, 0)
            
            # Initialize energy-aware collector
            collector = EnergyAwareCollector(
                network_env,
                self.config.get('controller_url', 'http://127.0.0.1:8080'),
                energy_model=self.energy_model
            )
            collector.set_network(net)
            
            # Pass the routing configuration to the collector
            if routing and 'paths' in routing:
                all_paths = routing['paths']
                sample_keys = random.sample(list(all_paths.keys()), min(10, len(all_paths)))
                sample_paths = {k: all_paths[k] for k in sample_keys}
                collector.set_configured_routes(sample_paths)
                print(f"  Configured {len(sample_paths)}/{len(all_paths)} route paths for collector (sampled)")

            # Initialize the drift injector
            drift_injector = DriftInjector(net, self.config.get('controller_url'))
            drift_injector.set_network(net)
            # Key: Let `drift_injector` know the current `routing_applier` and routing table,
            # This allows path drift to call `apply_partial_routes` and actually modify the routing table.
            drift_injector.set_routing_context(self.routing_applier, routing)
            

            samples = []
            stochastic_cfg = self.config.get('stochastic_drift', {})
            
            # First, estimate the total duration, then start the replayer to cover the entire experiment (baseline + main experiment).
            normal_pre_dur, normal_post_dur, event_schedule = self._build_event_schedule()
            baseline_dur = int(self.config.get('baseline_duration', 10))
            replay_duration = max(
                int(
                    baseline_dur +
                    normal_pre_dur + normal_post_dur +
                    sum(e['gap_before'] + e['settle_time'] + e['duration'] for e in event_schedule)
                ),
                30
            ) + int(stochastic_cfg.get('replay_tail_buffer', 10))
            
            print(f"    Event-driven schedule: {len(event_schedule)} drift events, replay_duration={replay_duration}s")
            for e in event_schedule:
                print(f"      event#{e['event_id']}: gap={e['gap_before']}s settle={e['settle_time']}s "
                      f"duration={e['duration']}s types={e['types']}")
            
            replayer.replay_with_iperf(scaled_tm, duration=replay_duration)
            time.sleep(5)
            
            # Collect baseline metrics
            baseline = self._collect_baseline_metrics(collector, duration=baseline_dur)
            
            # Use the baseline to generate an adaptive template and replace the original template
            adaptive_templates = self._compute_adaptive_intent_templates(baseline)
            self.intent_generator.INTENT_TEMPLATES = adaptive_templates
            print(f"    Adaptive intent templates installed:")
            for tname, t in adaptive_templates.items():
                print(f"      {tname}: delay∈[{t['delay_range'][0]:.0f},{t['delay_range'][1]:.0f}]ms, "
                      f"loss∈[{t['loss_range'][0]:.3f},{t['loss_range'][1]:.3f}], "
                      f"bw∈[{t['bandwidth_range'][0]:.2f},{t['bandwidth_range'][1]:.2f}]Mbps, "
                      f"power∈[{t['power_range'][0]:.0f},{t['power_range'][1]:.0f}]W, "
                      f"eff∈[{t['efficiency_range'][0]:.4f},{t['efficiency_range'][1]:.4f}]Mbps/W")
            
            # ============================================================
            # Generate Intent (Based on Adaptive Templates)
            # ============================================================
            hosts = net.hosts
            
            # First, perform a brief initial collection (3 seconds) to see which streams have actual traffic.
            # Then select the intended target from the streams with traffic
            print("    [pre-scan] Quick scan to find flows with traffic...")
            collector.start_collection(interval=1.0)
            time.sleep(3)
            pre_scan_snaps = collector.stop_collection()
            
            src_host = None
            dst_host = None
            path_nodes_for_intent = []
            
            if pre_scan_snaps:
                pre_snap_dict = pre_scan_snaps[-1].to_dict() if hasattr(pre_scan_snaps[-1], 'to_dict') else pre_scan_snaps[-1]
                observed_paths = pre_snap_dict.get('paths', {})
                
                # Find the stream with the highest throughput and the longest path
                best_flow = None
                best_score = -1
                for pid, pdata in observed_paths.items():
                    if not isinstance(pdata, dict):
                        continue
                    tput = float(pdata.get('e2e_throughput_mbps', 0) or 0)
                    loss = float(pdata.get('e2e_loss_rate', 0) or 0)
                    nodes = pdata.get('path_nodes', [])
                    nhops = len(nodes) - 1 if nodes else 0
                    
                    # Requirements: Sufficient traffic (>0.1 Mbps), low packet loss (<0.5%), and a path length of ≥4 hops
                    if tput > 0.1 and loss < 0.5 and nhops >= 4:
                        score = tput * nhops
                        if score > best_score:
                            best_score = score
                            best_flow = pdata
                
                if best_flow:
                    src_host = best_flow.get('src_host')
                    dst_host = best_flow.get('dst_host')
                    path_nodes_for_intent = best_flow.get('path_nodes', [])
                    print(f"    [pre-scan] Selected intent flow: {src_host}->{dst_host}, "
                          f"tput={best_flow.get('e2e_throughput_mbps',0):.2f}Mbps, "
                          f"path={path_nodes_for_intent}, hops={len(path_nodes_for_intent)-1}")
                else:
                    # Try again with less strict requirements
                    for pid, pdata in observed_paths.items():
                        if not isinstance(pdata, dict):
                            continue
                        tput = float(pdata.get('e2e_throughput_mbps', 0) or 0)
                        nodes = pdata.get('path_nodes', [])
                        nhops = len(nodes) - 1 if nodes else 0
                        if tput > 0.01 and nhops >= 3:
                            score = tput * nhops
                            if score > best_score:
                                best_score = score
                                best_flow = pdata
                    if best_flow:
                        src_host = best_flow.get('src_host')
                        dst_host = best_flow.get('dst_host')
                        path_nodes_for_intent = best_flow.get('path_nodes', [])
                        print(f"    [pre-scan] Selected (relaxed): {src_host}->{dst_host}, "
                              f"tput={best_flow.get('e2e_throughput_mbps',0):.4f}Mbps, "
                              f"path={path_nodes_for_intent}")
                    else:
                        print(f"    [pre-scan] Warning: No suitable flow found!")
                        # Printing all streams can help with diagnosis
                        for pid, pdata in observed_paths.items():
                            if isinstance(pdata, dict):
                                print(f"      {pid}: tput={pdata.get('e2e_throughput_mbps',0):.4f} "
                                      f"loss={pdata.get('e2e_loss_rate',0):.3f} "
                                      f"hops={len(pdata.get('path_nodes',[])) - 1}")
            
            # Fallback
            if not src_host or not dst_host:
                src_host = hosts[0].name if hosts else None
                dst_host = hosts[-1].name if len(hosts) > 1 else None
                print(f"    [pre-scan] Fallback to {src_host}->{dst_host}")
            
            # Retrieve the route from the routing table (if it wasn't obtained during pre-collection)
            if not path_nodes_for_intent:
                if routing and 'paths' in routing and src_host and dst_host:
                    all_paths = routing['paths']
                    forward_key = (src_host, dst_host)
                    reverse_key = (dst_host, src_host)
                    
                    if forward_key in all_paths:
                        matched = all_paths[forward_key]
                        if isinstance(matched, list):
                            path_nodes_for_intent = matched
                    elif reverse_key in all_paths:
                        matched = all_paths[reverse_key]
                        if isinstance(matched, list):
                            path_nodes_for_intent = list(reversed(matched))
                    
                    if path_nodes_for_intent:
                        print(f"    Using routing path for {src_host}->{dst_host}: {path_nodes_for_intent}")
            
            intents = self.intent_generator.generate_intent_batch(
                n=self.num_intents_per_experiment,
                src_host=src_host,
                dst_host=dst_host,
                path_nodes=path_nodes_for_intent,
            )
            
            print(f"  Generated {len(intents)} intents for this experiment:")
            for intent in intents:
                perf = intent.performance
                eng = intent.energy
                print(f"    {intent.intent_id} ({intent.intent_type.value}): "
                      f"delay≤{perf.delay.threshold:.0f}ms, "
                      f"loss≤{perf.loss.threshold:.3f}, "
                      f"power≤{eng.max_power.threshold:.0f}W, "
                      f"efficiency≥{eng.min_efficiency.threshold:.4f}Mbps/W")
            
            print("    [timeline] Collecting randomized normal pre-drift samples...")
            normal_samples, normal_snapshots = self._collect_phase_samples(
                collector, intents, normal_pre_dur,
                topo_name=topo_name,
                sched_config_id=sched_config['id'],
                injected_drift_type='normal',
                baseline_routing=routing,
            )
            samples.extend(normal_samples)

            if normal_snapshots:
                collector.set_baseline(normal_snapshots[0])
            

            if normal_samples and intents:
                first_normal = normal_samples[0]
                observed_paths = first_normal.get('paths', {})
                for intent in intents:
                    if not intent.path or not intent.path.waypoints:
                        continue
                    
                    i_match = getattr(intent, 'match', None) or {}
                    if not isinstance(i_match, dict):
                        i_match = {}
                    i_src = i_match.get('src', '')
                    i_dst = i_match.get('dst', '')
                    
                    for pid, pdata in observed_paths.items():
                        src_h = pdata.get('src_host', '')
                        dst_h = pdata.get('dst_host', '')
                        if ((src_h == i_src and dst_h == i_dst) or
                            (src_h == i_dst and dst_h == i_src)):
                            actual_nodes = pdata.get('path_nodes', [])
                            if actual_nodes and len(actual_nodes) >= 3:
                                # Check whether the current waypoints are on the actual route
                                old_wp = intent.path.waypoints
                                missing = [wp for wp in old_wp if wp not in actual_nodes]
                                if missing:
                                    # Reselect waypoints starting from the middle node of the actual path
                                    middle = actual_nodes[1:-1]
                                    if middle:
                                        n_wp = min(len(old_wp), len(middle))
                                        new_wp_indices = np.random.choice(
                                            len(middle), size=n_wp, replace=False)
                                        new_wp = [middle[int(i)] for i in new_wp_indices]
                                        print(f"    [waypoint fix] Intent {intent.intent_id}: "
                                              f"old waypoints {old_wp} not in actual path {actual_nodes}, "
                                              f"replaced with {new_wp}")
                                        intent.path.waypoints = new_wp
                                    else:
                                        print(f"    [waypoint fix] Intent {intent.intent_id}: "
                                              f"actual path too short, clearing waypoints")
                                        intent.path.waypoints = []
                            break
            

            if normal_samples and intents:
                print("    [threshold calibration] Calibrating thresholds from normal samples...")
                
                for intent in intents:
                    i_match = getattr(intent, 'match', None) or {}
                    if not isinstance(i_match, dict):
                        i_match = {}
                    i_src = i_match.get('src', '')
                    i_dst = i_match.get('dst', '')
                    
                    # Collect all observations for the intent-matching stream during the normal phase
                    obs_delays = []
                    obs_losses = []
                    obs_tputs = []
                    obs_hops = []
                    obs_powers = []
                    obs_effs = []
                    
                    for samp in normal_samples:
                        paths_data = samp.get('paths', {})
                        for pid, pd in paths_data.items():
                            if not isinstance(pd, dict):
                                continue
                            sh = pd.get('src_host', '')
                            dh = pd.get('dst_host', '')
                            if ((sh == i_src and dh == i_dst) or
                                (sh == i_dst and dh == i_src)):
                                d = float(pd.get('e2e_delay_ms', 0) or 0)
                                l = float(pd.get('e2e_loss_rate', 0) or 0)
                                t = float(pd.get('e2e_throughput_mbps', 0) or 0)
                                h = int(pd.get('num_hops', 0) or 0)
                                if d > 0: obs_delays.append(d)
                                obs_losses.append(l)
                                obs_tputs.append(t)
                                if h > 0: obs_hops.append(h)
                                break
                        
                        pw = float(samp.get('total_power_watts', 0) or 0)
                        ef = float(samp.get('energy_efficiency', 0) or 0)
                        if pw > 0: obs_powers.append(pw)
                        if ef > 0: obs_effs.append(ef)
                    
                    if not obs_tputs:
                        print(f"      No observations for {i_src}->{i_dst}, skipping calibration")
                        continue
                    
                    calibrated = []
                    
                    # delay: threshold >= observed P99 × 2
                    if intent.performance and intent.performance.delay and obs_delays:
                        old_th = intent.performance.delay.threshold
                        p99 = float(np.percentile(obs_delays, 99))
                        new_th = max(old_th, p99 * 20)
                        if new_th != old_th:
                            intent.performance.delay.threshold = new_th
                            calibrated.append(f"delay: {old_th:.1f}->{new_th:.1f}ms")
                    
                    # loss: Show loss only when there is traffic
                    if intent.performance and intent.performance.loss:
                        old_th = intent.performance.loss.threshold
                        valid_losses = [obs_losses[i] for i in range(len(obs_losses))
                                       if i < len(obs_tputs) and obs_tputs[i] > 0.01]
                        if valid_losses:
                            p99 = float(np.percentile(valid_losses, 99))
                            new_th = max(old_th, p99 * 20, 1.5)
                        else:
                            new_th = max(old_th, 1.5)
                        if new_th != old_th:
                            intent.performance.loss.threshold = new_th
                            calibrated.append(f"loss: {old_th:.4f}->{new_th:.4f}")
                    
                    # bandwidth: threshold <= measured effective throughput P1 × 0.3
                    if intent.performance and intent.performance.bandwidth and obs_tputs:
                        old_th = intent.performance.bandwidth.threshold
                        valid_tputs = [t for t in obs_tputs if t > 0.001]
                        if valid_tputs:
                            p1 = float(np.percentile(valid_tputs, 1))
                            new_th = min(old_th, p1 * 0.05)
                        else:
                            new_th = min(old_th, 0.0001)
                        if new_th != old_th:
                            intent.performance.bandwidth.threshold = new_th
                            calibrated.append(f"bw: {old_th:.4f}->{new_th:.6f}Mbps")
                    
                    # power: threshold >= observed P99 × 1.1
                    if intent.energy and intent.energy.max_power and obs_powers:
                        old_th = intent.energy.max_power.threshold
                        p99 = float(np.percentile(obs_powers, 99))
                        new_th = max(old_th, p99 * 1.3)
                        if new_th != old_th:
                            intent.energy.max_power.threshold = new_th
                            calibrated.append(f"power: {old_th:.0f}->{new_th:.0f}W")
                    
                    # efficiency: threshold <= actual P1 × 0.3
                    if intent.energy and intent.energy.min_efficiency and obs_effs:
                        old_th = intent.energy.min_efficiency.threshold
                        p1 = float(np.percentile(obs_effs, 1))
                        new_th = min(old_th, p1 * 0.05)
                        if new_th != old_th:
                            intent.energy.min_efficiency.threshold = new_th
                            calibrated.append(f"eff: {old_th:.4f}->{new_th:.6f}")
                    
                    # max_hops: threshold >= observed max + 2
                    if intent.path and intent.path.max_hops and obs_hops:
                        old_th = intent.path.max_hops
                        max_obs = max(obs_hops)
                        new_th = max(old_th, max_obs + 5)
                        if new_th != old_th:
                            intent.path.max_hops = new_th
                            calibrated.append(f"hops: {old_th}->{new_th}")
                    
                    if calibrated:
                        print(f"      Intent {intent.intent_id}: {', '.join(calibrated)}")
                    else:
                        print(f"      Intent {intent.intent_id}: no calibration needed")
                
                # Relabel the samples from the normal phase after calibration
                print("    [re-label] Re-labeling normal samples with calibrated thresholds...")
                normal_samples_relabeled = []
                for intent in intents:
                    relabeled = self._label_snapshots(
                        normal_snapshots, intent,
                        injected_drift_type='normal',
                        topo_name=topo_name,
                        sched_config_id=sched_config['id'],
                        baseline_routing=routing,
                    )
                    normal_samples_relabeled.extend(relabeled)
                
                # Replace
                samples = [s for s in samples if s not in normal_samples]
                normal_samples = normal_samples_relabeled
                samples.extend(normal_samples)
                
                n_cal_drift = sum(1 for s in normal_samples 
                                 if any(s.get('clause_labels', {}).get(k, 0) for k in ['perf', 'path', 'energy']))
                print(f"    [re-label] After calibration: {len(normal_samples)} normal samples, "
                      f"{n_cal_drift} drift ({n_cal_drift/max(len(normal_samples),1):.1%})")

            for event in event_schedule:
                gap_before = int(event.get('gap_before', 0))
                if gap_before > 0:
                    print(f"    [timeline] Collecting pre-event normal gap: {gap_before}s")
                    gap_samples, _ = self._collect_phase_samples(
                        collector, intents, gap_before,
                        topo_name=topo_name,
                        sched_config_id=sched_config['id'],
                        injected_drift_type='normal',
                        baseline_routing=routing,
                    )
                    samples.extend(gap_samples)

                links = drift_injector._get_all_links()
                if not links:
                    continue

                injected_type, drift_location, drift_params, extra_energy_detection = self._inject_event_drifts(
                    event, drift_injector, links, topo_data, routing,
                    intents=intents,
                )

                settle_time = int(event.get('settle_time', 0))
                if settle_time > 0:
                    time.sleep(settle_time)

                phase_samples, _ = self._collect_phase_samples(
                    collector, intents, int(event.get('duration', 0)),
                    topo_name=topo_name,
                    sched_config_id=sched_config['id'],
                    injected_drift_type=injected_type,
                    drift_location=drift_location,
                    drift_params=drift_params,
                    extra_energy_detection=extra_energy_detection,
                    baseline_routing=routing,
                )
                samples.extend(phase_samples)

                drift_injector.clear_all_drifts()

            print("    [timeline] Collecting randomized normal post-drift samples...")
            normal_post_samples, _ = self._collect_phase_samples(
                collector, intents, normal_post_dur,
                topo_name=topo_name,
                sched_config_id=sched_config['id'],
                injected_drift_type='normal',
                baseline_routing=routing,
            )
            samples.extend(normal_post_samples)

            # Clean up
            try:
                replayer.stop_replay()
            except Exception as e:
                print(f"Warning: Error stopping replayer:{e}")
            
            # Record routing information for all samples
            for sample in samples:
                sample['routing_type'] = routing.get('type', 'unknown')
                sample['routing_id'] = routing.get('id', 0)
                sample['routing_verified'] = routing_verification.get('is_valid', False) if routing_verification else False
                sample['routing_issues'] = len(routing_verification.get('issues', [])) if routing_verification else -1
                sample['configured_routes_count'] = len(routing_verification.get('configured_routes', {})) if routing_verification else 0
                sample['installed_flows_count'] = len(routing_verification.get('installed_flows', [])) if routing_verification else 0
                
        except Exception as e:
            print(f"Error:{e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clear the router configuration
            try:
                self.routing_applier.clear_routes()
            except Exception as e:
                print(f"Warning: Error clearing routes:{e}")
                
            try:
                if hasattr(self, 'traffic_gen') and hasattr(self.traffic_gen, 'stop_all_traffic'):
                    self.traffic_gen.stop_all_traffic()
                for host in network_env.net.hosts:
                    p = host.popen(['killall', '-9', 'iperf', 'iperf3', 'ping', 'tcpreplay'])
                    p.wait()
                time.sleep(1)
            except Exception as e:
                print(f"Warning: Error stopping traffic processes: {e}")
                
            try:
                network_env.stop()
            except Exception as e:
                print(f"Warning: Error stopping network:{e}")
        
        print(f"Returning {len(samples)} samples")
        return samples
    
    def _extract_link_info(self, net):
        """Extracting link information from the Mininet network"""
        links = []
        for link in net.links:
            intf1, intf2 = link.intf1, link.intf2
            node1, node2 = intf1.node, intf2.node
        
            # Focus only on the links between switches
            if hasattr(node1, 'dpid') and hasattr(node2, 'dpid'):
                links.append({
                    'src': node1.name,
                    'dst': node2.name,
                    'src_port': node1.ports[intf1],
                    'dst_port': node2.ports[intf2]
                })
        return links
    
    def _apply_scheduling_config(self, qos_config, sched_config):
        """Application Scheduling Configuration"""
        net = qos_config.network_env.net
        
        for switch in net.switches:
            node_conf = sched_config['nodes'].get(switch.name, {})
            policy = node_conf.get('policy', 'FIFO')
            queue_size = node_conf.get('queue_size', 16000)
            weights = node_conf.get('weights')
            
            for intf in switch.intfList():
                if intf.name != 'lo' and not intf.name.startswith('lo'):
                    # Clear existing configuration
                    switch.cmd(f'tc qdisc del dev {intf.name} root 2>/dev/null')
                    
                    # Set the queue size
                    switch.cmd(f'ip link set {intf.name} txqueuelen {queue_size // 8000}')
                    
                    if policy == 'FIFO':
                        switch.cmd(f'tc qdisc add dev {intf.name} root pfifo limit {queue_size // 1000}')
                    elif policy == 'SP':
                        switch.cmd(f'tc qdisc add dev {intf.name} root prio bands 3')
                    elif policy == 'WFQ' and weights:
                        switch.cmd(f'tc qdisc add dev {intf.name} root handle 1: htb default 30')
                        switch.cmd(f'tc class add dev {intf.name} parent 1: classid 1:1 htb rate 100mbit')
                        switch.cmd(f'tc class add dev {intf.name} parent 1:1 classid 1:10 htb rate {int(weights["tos0"])}mbit')
                        switch.cmd(f'tc class add dev {intf.name} parent 1:1 classid 1:20 htb rate {int(weights["tos1"])}mbit')
                        switch.cmd(f'tc class add dev {intf.name} parent 1:1 classid 1:30 htb rate {int(weights["tos2"])}mbit')
                    elif policy == 'DRR':
                        switch.cmd(f'tc qdisc add dev {intf.name} root sfq perturb 10')
    
    def run(self):
        """Run full dataset generation"""
        print("=" * 60)
        print("Real Trace Dataset Generation (v2 - Intent-Driven)")
        print("=" * 60)
        print(f"Drift types: normal(0), performance(1), path(2), energy(3)")
        print(f"Intents per experiment: {self.num_intents_per_experiment}")
        print(f"Labels: AUTO-DERIVED from intent constraints")
        print("=" * 60)
        
        # Prepare the traffic configuration
        print("\n[1/5] Preparing traffic profile...")
        self.prepare_traffic_profile()
        
        # Prepare the topology
        print("\n[2/5] Preparing topologies...")
        topologies = self.prepare_topologies()
        
        if not topologies:
            print("Error: No topologies available")
            return
        
        # Generate scheduling configuration
        print("\n[3/5] Generating scheduling configurations...")
        sched_configs = self.generate_scheduling_configs(
            self.config.get('num_scheduling_configs', 10)
        )
        
        # Conduct the experiment
        print("\n[4/5] Running experiments...")
        all_samples = []
        output_path1 = os.path.join(self.output_dir, 'samples_raw_new.jsonl')
        # Important: Delete the old `samples_raw_new.jsonl` file to avoid mixing it with samples from previous experiments.
        open(output_path1, 'w').close()
        print(f"    Cleared old samples file: {output_path1}")
        available_topos = list(topologies.keys())
        
        for topo_name in available_topos:
            topo_data = topologies[topo_name]
            print(f"\n  Topology: {topo_name}")
            print(f"    Nodes: {topo_data.get('num_nodes', 'N/A')}, "
                  f"Links: {topo_data.get('num_links', 'N/A')}")
            
            # Generate route variants
            routing_gen = RoutingGenerator(
                self._save_temp_topology(topo_data['config'])
            )
            routings = routing_gen.generate_routing_variants(
                num_variants=min(self.NUM_ROUTING_VARIANTS, self.config.get('num_routings', 5))
            )
            
            # Select specific scheduling configurations
            selected_scheds = sched_configs[:self.config.get('num_scheduling_configs', 5)]
            
            for routing_idx, routing in enumerate(routings[:self.config.get('num_routings', 3)]):
                for sched_config in selected_scheds[:self.config.get('num_scheduling_configs', 2)]:
                    print(f"    Routing {routing_idx}, Sched {sched_config['id']}...")
                    try:
                        samples = self.run_single_experiment(
                            topo_name,
                            topo_data,
                            routing,
                            sched_config,
                            topo_data.get('traffic_matrix')
                        )
                        all_samples.extend(samples)
                        print(f"      Collected {len(samples)} samples")
                        with open(output_path1, 'a') as f:
                            for s in samples:
                                f.write(json.dumps(s, default=str) + '\n')
                        print(f"      Already {len(samples)} tokens, total {len(all_samples)}")
                    except Exception as e:
                        print(f"      Error: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
        
        # Save the dataset
        print("\n[5/5] Saving dataset...")
        self._save_dataset(all_samples)
        
        print("\n" + "=" * 60)
        print("Dataset Generation Complete!")
        print("=" * 60)
        print(f"Total samples: {len(all_samples)}")
        print(f"Output directory: {self.output_dir}")
        
        # Label Distribution
        label_counts = {}
        for s in all_samples:
            dt = s.get('drift_type', 'unknown')
            label_counts[dt] = label_counts.get(dt, 0) + 1
        
        print("\nLabel distribution (auto-derived from intents):")
        for dt, count in sorted(label_counts.items()):
            pct = count / len(all_samples) * 100 if all_samples else 0
            print(f"  {dt}: {count} ({pct:.1f}%)")
        
        # Statistics on Print Intent Diversity
        intent_ids = set()
        for s in all_samples:
            intent = s.get('intent', {})
            if intent:
                intent_ids.add(intent.get('intent_id', ''))
        print(f"\nUnique intents used: {len(intent_ids)}")
        
        # Print statistics for “same status, different labels”
        self._print_label_diversity_stats(all_samples)
        
        return all_samples
    
    def _print_label_diversity_stats(self, samples):

        # Group by timestamp
        by_timestamp = {}
        for s in samples:
            ts = s.get('timestamp', 0)
            ts_key = f"{ts:.2f}"
            if ts_key not in by_timestamp:
                by_timestamp[ts_key] = set()
            by_timestamp[ts_key].add(s.get('drift_label', -1))
        
        multi_label_count = sum(1 for labels in by_timestamp.values() if len(labels) > 1)
        total_timestamps = len(by_timestamp)
        
        if total_timestamps > 0:
            print(f"\nLabel diversity (same state, different intents → different labels):")
            print(f"  Timestamps with multiple labels: {multi_label_count}/{total_timestamps} "
                  f"({multi_label_count/total_timestamps*100:.1f}%)")
    
    def _build_sequences(self, samples, window_size=10, horizon=3, min_persist=2,
                         max_gap_seconds=5.0):

        from collections import defaultdict
        
        # Grouped by (experimental group + intention)
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
        n_filtered_gap = 0  # Number of windows filtered due to time breaks
        
        for group_key, group_samples in groups.items():
            group_samples.sort(key=lambda x: x.get('timestamp', 0))
            
            n = len(group_samples)
            if n < window_size + horizon:
                continue
            
            for i in range(n - window_size - horizon + 1):
                window = group_samples[i : i + window_size]
                future = group_samples[i + window_size : i + window_size + horizon]
                
                # ============================================================
                # Time breakpoint check: If the interval between adjacent snapshots within a window+future exceeds the threshold,
                # Indicates physical discontinuity; discard this window
                # ============================================================
                full_seq = window + future
                has_gap = False
                for k in range(1, len(full_seq)):
                    gap = full_seq[k].get('timestamp', 0) - full_seq[k-1].get('timestamp', 0)
                    if gap > max_gap_seconds or gap < 0:
                        has_gap = True
                        break
                
                if has_gap:
                    n_filtered_gap += 1
                    continue
                
                # ============================================================
                # Clause-level persistence-aware labeling
                # Check each clause individually to see if there has been a default for min_persist consecutive time steps
                # ============================================================
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
                
                # Backward compatibility: Single `future_label` (priority: path > perf > energy)
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
                
                # Current time (last step of the window)
                current = window[-1]
                
                seq_sample = {
                    'window': window,
                    'intent': current.get('intent', {}),
                    'future_clause_labels': future_clause_labels,
                    'future_has_any_drift': future_has_any_drift,
                    'future_label': future_label,
                    'current_clause_labels': current.get('clause_labels', 
                        {'perf': 0, 'path': 0, 'energy': 0}),
                    'experiment_id': f"{group_key[0]}_{group_key[1]}_{group_key[2]}",
                    'intent_id': group_key[3],
                    'window_start_ts': window[0].get('timestamp', 0),
                    'window_end_ts': window[-1].get('timestamp', 0),
                    'drift_location': None,
                    # The SAFLA-style baseline requires: inheritance from the snapshot within the window baseline_routing_paths
                    'baseline_routing_paths': window[0].get('baseline_routing_paths', {}),
                }
                
                # Drift location information: Get the position of the first clause violation in the future
                # If it's not in the future, fall back to searching at the end of the window (because drift may occur at the end of the window,
                # and the future snapshot might happen to fall within the normal segment)
                if future_has_any_drift:
                    for f in future:
                        cl = f.get('clause_labels', {})
                        if any(cl.values()) and f.get('drift_location'):
                            seq_sample['drift_location'] = f['drift_location']
                            break
                    
                    # Fall back: If not found in future, search in the window (from the most recent snapshot backward)
                    if not seq_sample['drift_location']:
                        for f in reversed(window):
                            cl = f.get('clause_labels', {})
                            if any(cl.values()) and f.get('drift_location'):
                                seq_sample['drift_location'] = f['drift_location']
                                break
                
                sequence_samples.append(seq_sample)
        
        if n_filtered_gap > 0:
            print(f"  Filtered {n_filtered_gap} windows due to time gaps "
                  f"(>{max_gap_seconds}s between adjacent snapshots)")
        
        return sequence_samples
    
    def _split_raw_samples_temporally(self, samples):
        from collections import defaultdict

        groups = defaultdict(list)
        for s in samples:
            group_key = (
                s.get('topology', 'unknown'),
                s.get('routing_id', 0),
                s.get('scheduling_config', 0),
                s.get('intent', {}).get('intent_id', 'unknown'),
            )
            groups[group_key].append(s)

        train_raw, val_raw, test_raw = [], [], []
        for _, group in groups.items():
            group.sort(key=lambda x: x.get('timestamp', 0))
            n = len(group)
            if n == 1:
                train_raw.extend(group)
                continue
            train_end = max(int(n * 0.70), 1)
            val_end = max(int(n * 0.85), train_end + 1)
            val_end = min(val_end, n)
            train_raw.extend(group[:train_end])
            val_raw.extend(group[train_end:val_end])
            test_raw.extend(group[val_end:])

        return train_raw, val_raw, test_raw

    def _save_dataset(self, samples):

        window_size = self.config.get('window_size', 10)
        horizon = self.config.get('prediction_horizon', 3)
        min_persist = self.config.get('min_persist', 2)
        max_gap = self.config.get('max_gap_seconds', 12.0)

        print(f"\nBuilding sequence samples...")
        print(f"  Window size: {window_size}")
        print(f"  Prediction horizon: {horizon}")
        print(f"  Min persistence: {min_persist}")
        print(f"  Max gap between snapshots: {max_gap}s")

        # 1. First, construct a sliding window over all raw samples
        all_seq_samples = self._build_sequences(
            samples, window_size, horizon, min_persist, max_gap_seconds=max_gap
        )
        print(f"  Generated {len(all_seq_samples)} sequence samples from {len(samples)} raw snapshots")

        if not all_seq_samples:
            print("Warning: No sequence samples generated. Saving raw snapshots instead.")
            self._save_dataset_fallback(samples)
            return

        # 2. Group by experiment_id, sort within each group by window_end_ts, and perform temporal split
        # Leave a boundary_gap at the split points to prevent boundary window leakage
        from collections import defaultdict
        seq_groups = defaultdict(list)
        for s in all_seq_samples:
            seq_groups[s['experiment_id']].append(s)
        
        boundary_gap = window_size + horizon
        # What is the minimum number of sequences required per group for splitting? (If there are too few, they are all assigned to the training set.)
        min_seqs_for_split = self.config.get('min_seqs_for_split', 10)
        train_samples, val_samples, test_samples = [], [], []
        n_skipped_groups = 0
        
        for exp_id, exp_seqs in seq_groups.items():
            exp_seqs.sort(key=lambda x: x['window_end_ts'])
            n = len(exp_seqs)
            
            if n < min_seqs_for_split:
                # The batch size is too small; use it all for training (we'll compensate for this later).
                train_samples.extend(exp_seqs)
                n_skipped_groups += 1
                continue
            
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            
            # Adaptive boundary_gap：
            # - When the group is large enough (≥ 3*boundary_gap), leave a strict gap to prevent boundary leakage
            # - When the group is small, leave a 0 gap and allow slight boundary overlap (which is better than having an empty set of validation and test data)
            if n >= 3 * boundary_gap:
                effective_gap = boundary_gap
            else:
                effective_gap = 0
            
            val_start = min(train_end + effective_gap, val_end)
            test_start = min(val_end + effective_gap, n)
            
            train_samples.extend(exp_seqs[:train_end])
            if val_start < val_end:
                val_samples.extend(exp_seqs[val_start:val_end])
            if test_start < n:
                test_samples.extend(exp_seqs[test_start:])

        total_seq = len(train_samples) + len(val_samples) + len(test_samples)
        print(f"  After temporal split -> train:{len(train_samples)} val:{len(val_samples)} test:{len(test_samples)}")
        if n_skipped_groups > 0:
            print(f"  Note: {n_skipped_groups}/{len(seq_groups)} groups too small (<{min_seqs_for_split} seqs), "
                  f"their sequences all went to train")

        if total_seq == 0:
            print("Warning: All sequence samples filtered out. Saving raw snapshots instead.")
            self._save_dataset_fallback(samples)
            return
        
        # If `val` or `test` is still empty (all groups have been skipped), perform a global fallback partitioning
        if len(val_samples) == 0 or len(test_samples) == 0:
            print(f"  Warning: val or test is empty after per-group split. "
                  f"Performing global fallback split.")
            # Sort globally by window_end_ts
            all_seq_samples.sort(key=lambda x: x['window_end_ts'])
            N = len(all_seq_samples)
            t_end = int(N * 0.70)
            v_end = int(N * 0.85)
            train_samples = all_seq_samples[:t_end]
            val_samples = all_seq_samples[t_end:v_end]
            test_samples = all_seq_samples[v_end:]
            print(f"  Global split -> train:{len(train_samples)} val:{len(val_samples)} test:{len(test_samples)}")

        train_path = os.path.join(self.output_dir, 'train_T8.json')
        val_path = os.path.join(self.output_dir, 'val_T8.json')
        test_path = os.path.join(self.output_dir, 'test_T8.json')

        with open(train_path, 'w') as f:
            json.dump(train_samples, f, indent=2, default=str)
        print(f"Saved {len(train_samples)} training samples to {train_path}")

        with open(val_path, 'w') as f:
            json.dump(val_samples, f, indent=2, default=str)
        print(f"Saved {len(val_samples)} validation samples to {val_path}")

        with open(test_path, 'w') as f:
            json.dump(test_samples, f, indent=2, default=str)
        print(f"Saved {len(test_samples)} test samples to {test_path}")

        config_path = os.path.join(self.output_dir, 'generation_config_new.json')
        with open(config_path, 'w') as f:
            json.dump({
                'config': self.config,
                'drift_types': self.DRIFT_TYPES,
                'labeling_method': 'auto_derived_from_intent',
                'prediction_mode': True,
                'sequence_build_after_temporal_split': True,
                'window_size': window_size,
                'prediction_horizon': horizon,
                'min_persist': min_persist,
                'num_intents_per_experiment': self.num_intents_per_experiment,
                'intent_templates': list(IntentGenerator.INTENT_TEMPLATES.keys()),
                'has_baseline_routing_paths': True,  # SAFLA baseline support tags
                'dataset_split': {
                    'train': len(train_samples),
                    'val': len(val_samples),
                    'test': len(test_samples)
                }
            }, f, indent=2, default=str)

        print("\nClause-level label distribution (future):")
        for name, data in [('Train', train_samples), ('Val', val_samples), ('Test', test_samples)]:
            if not data:
                print(f"  {name}: empty")
                continue

            n_total = len(data)
            n_perf = sum(1 for s in data if s.get('future_clause_labels', {}).get('perf', 0))
            n_path = sum(1 for s in data if s.get('future_clause_labels', {}).get('path', 0))
            n_energy = sum(1 for s in data if s.get('future_clause_labels', {}).get('energy', 0))
            n_any = sum(1 for s in data if s.get('future_has_any_drift', False))
            n_normal = n_total - n_any
            n_mixed = sum(1 for s in data if sum(s.get('future_clause_labels', {}).values()) >= 2)

            print(f"  {name}: {n_total} total | "
                  f"normal:{n_normal} ({n_normal/n_total*100:.0f}%) "
                  f"perf:{n_perf} path:{n_path} energy:{n_energy} mixed:{n_mixed}")

    def _save_dataset_fallback(self, samples):
        """Fallback plan: Instead of random partitioning, save the original snapshots in chronological order."""
        train_raw, val_raw, test_raw = self._split_raw_samples_temporally(samples)

        for name, data, fname in [('Train', train_raw, 'train_new.json'),
                                   ('Val', val_raw, 'val_new.json'),
                                   ('Test', test_raw, 'test_new.json')]:
            path = os.path.join(self.output_dir, fname)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"Saved {len(data)} {name} raw snapshot samples (temporal fallback)")

    def _save_temp_topology(self, topo_config):
        """Save the temporary topology configuration"""
        temp_path = os.path.join(self.output_dir, 'temp_topology.yaml')
        with open(temp_path, 'w') as f:
            yaml.dump(topo_config, f)
        return temp_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate dataset with real traffic traces (v2 - Intent-Driven)')
    parser.add_argument('--config', type=str, default='configs/real_trace_experiment.yaml')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--num-intents', type=int, default=3, help='Number of intents per experiment')
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'output_dir': 'data/real_trace_dataset',
        'controller_ip': '127.0.0.1',
        'controller_port': 6653,
        'controller_url': 'http://127.0.0.1:8080',
        'sndlib_dir': 'data/real_traces/sndlib',
        'traffic_profile_path': None,
        'max_packets': 50000,
        'time_scale_factor': 1.0,
        'max_traffic_rate': 10.0,
        'num_routings': 5,
        'num_scheduling_configs': 5,
        'normal_duration': 30,
        'drift_samples_per_config': 3,
        'stochastic_drift': {
            'normal_pre_min': 20,
            'normal_pre_max': 45,
            'normal_post_min': 15,
            'normal_post_max': 40,
            'drift_duration_scale': 8.0,
            'drift_duration_offset': 2.0,
            'drift_duration_min': 3.0,
            'drift_duration_max': 30.0,
            'inter_drift_gap_scale': 8.0,
            'inter_drift_gap_min': 2.0,
            'inter_drift_gap_max': 25.0,
            'episodes_choices': [1, 2, 3, 4],
            'episodes_probs': [0.3, 0.4, 0.2, 0.1],
            'delay_lognormal_mean': 3.5,
            'delay_lognormal_sigma': 0.8,
            'delay_min': 10,
            'delay_max': 200,
            'loss_beta_a': 2.0,
            'loss_beta_b': 20.0,
            'loss_min': 0.001,
            'loss_max': 0.20,
            'mixed_drift_probability': 0.15,
            'secondary_energy_target_size': 2,
            'settle_time_min': 1,
            'settle_time_max': 3,
            'recovery_collect_threshold': 3,
            'replay_tail_buffer': 10
        },
        
        # Power consumption settings
        'intent_max_power': 1500.0,
        'energy': {
            'switch_chassis_power': 100.0,
            'switch_idle_port_power': 2.0,
            'switch_active_port_power': 5.0,
            'switch_dynamic_power': 10.0,
            'link_transceiver_power': 1.0,
            'link_dynamic_power': 0.5
        },
        
        # Intent Generation Configuration (New)
        'num_intents_per_experiment': args.num_intents,
        'intent_seed': 42,
        
        'drift_distribution': {
            'normal': 0.4,
            'performance': 0.2,
            'path': 0.2,
            'energy': 0.2
        }
    }
    
    # Load configuration file
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = yaml.safe_load(f)
            if file_config:
                config.update(file_config)
    
    # Command-line arguments override
    if args.output:
        config['output_dir'] = args.output
    
    # Quick test mode
    if args.quick:
        config['num_routings'] = 1
        config['num_scheduling_configs'] = 1
        config['normal_duration'] = 10
        config['drift_samples_per_config'] = 1
        config['stochastic_drift']['normal_pre_min'] = 8
        config['stochastic_drift']['normal_pre_max'] = 12
        config['stochastic_drift']['normal_post_min'] = 6
        config['stochastic_drift']['normal_post_max'] = 10
        config['stochastic_drift']['drift_duration_min'] = 2
        config['stochastic_drift']['drift_duration_max'] = 6
        config['stochastic_drift']['inter_drift_gap_min'] = 1
        config['stochastic_drift']['inter_drift_gap_max'] = 4
        config['stochastic_drift']['episodes_choices'] = [1, 2]
        config['stochastic_drift']['episodes_probs'] = [0.7, 0.3]
        config['output_dir'] = config['output_dir'] + '_quick'
    
    print("=" * 60)
    print("Real Trace Dataset Generator (v2 - Intent-Driven)")
    print("=" * 60)
    print(f"\nIntents per experiment: {config['num_intents_per_experiment']}")
    print(f"Labeling method: AUTO-DERIVED from intent constraints")
    print(f"\nPlease ensure Ryu controller is running:")
    print("  cd ~/intent_drift_platform_new2_realtrace")
    print("  source ~/ryu_env_py310/bin/activate")
    print("  ryu-manager ryu_controller/intent_controller.py --ofp-tcp-listen-port 6653 --wsapi-port 8080")
    print("=" * 60)
    
    input("\nPress Enter when ready...")
    
    generator = RealTraceDatasetGenerator(config)
    generator.run()


if __name__ == '__main__':
    main()
