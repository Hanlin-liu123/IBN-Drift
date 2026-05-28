# data_collection/collector.py


import time
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import subprocess
import re
import json
import networkx as nx

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.energy_model import (
    NetworkEnergyModel, EnergyMetrics, EnergyDriftDetector,
    create_default_energy_model
)


@dataclass
class LinkMetrics:
    """Link-level metrics"""
    link_id: str
    src: str
    dst: str
    timestamp: float = 0.0

    tx_bytes: int = 0
    rx_bytes: int = 0
    tx_packets: int = 0
    rx_packets: int = 0

    delay_ms: float = 0.0
    jitter_ms: float = 0.0
    loss_rate: float = 0.0
    throughput_mbps: float = 0.0
    utilization: float = 0.0

    delay_p10: float = 0.0
    delay_p20: float = 0.0
    delay_p50: float = 0.0
    delay_p80: float = 0.0
    delay_p90: float = 0.0
    delay_p99: float = 0.0

    power_watts: float = 0.0

    def to_dict(self) -> dict:
        return {
            'link_id': self.link_id,
            'src': self.src,
            'dst': self.dst,
            'timestamp': self.timestamp,
            'tx_bytes': self.tx_bytes,
            'rx_bytes': self.rx_bytes,
            'delay_ms': self.delay_ms,
            'jitter_ms': self.jitter_ms,
            'loss_rate': self.loss_rate,
            'throughput_mbps': self.throughput_mbps,
            'utilization': self.utilization,
            'delay_p10': self.delay_p10,
            'delay_p20': self.delay_p20,
            'delay_p50': self.delay_p50,
            'delay_p80': self.delay_p80,
            'delay_p90': self.delay_p90,
            'delay_p99': self.delay_p99,
            'power_watts': self.power_watts
        }


@dataclass
class SwitchMetrics:
    """Switch-level metrics"""
    switch_id: str
    timestamp: float = 0.0

    port_stats: Dict[int, dict] = field(default_factory=dict)
    num_active_ports: int = 0

    total_throughput_mbps: float = 0.0
    total_packets: int = 0

    power_watts: float = 0.0
    cpu_utilization: float = 0.0

    def to_dict(self) -> dict:
        return {
            'switch_id': self.switch_id,
            'timestamp': self.timestamp,
            'num_active_ports': self.num_active_ports,
            'total_throughput_mbps': self.total_throughput_mbps,
            'total_packets': self.total_packets,
            'power_watts': self.power_watts,
            'cpu_utilization': self.cpu_utilization
        }


@dataclass
class PathMetrics:
    """Path-level metrics"""
    path_id: str
    src_host: str
    dst_host: str
    path_nodes: List[str] = field(default_factory=list)
    timestamp: float = 0.0

    e2e_delay_ms: float = 0.0
    e2e_jitter_ms: float = 0.0
    e2e_loss_rate: float = 0.0
    e2e_throughput_mbps: float = 0.0

    path_power_watts: float = 0.0
    num_hops: int = 0

    def to_dict(self) -> dict:
        return {
            'path_id': self.path_id,
            'src_host': self.src_host,
            'dst_host': self.dst_host,
            'path_nodes': self.path_nodes,
            'timestamp': self.timestamp,
            'e2e_delay_ms': self.e2e_delay_ms,
            'e2e_jitter_ms': self.e2e_jitter_ms,
            'e2e_loss_rate': self.e2e_loss_rate,
            'e2e_throughput_mbps': self.e2e_throughput_mbps,
            'path_power_watts': self.path_power_watts,
            'num_hops': self.num_hops
        }


@dataclass
class NetworkSnapshot:
    """Network state snapshot"""
    timestamp: float = 0.0

    link_metrics: Dict[str, LinkMetrics] = field(default_factory=dict)
    switch_metrics: Dict[str, SwitchMetrics] = field(default_factory=dict)
    path_metrics: Dict[str, PathMetrics] = field(default_factory=dict)

    total_throughput_mbps: float = 0.0
    avg_delay_ms: float = 0.0
    avg_loss_rate: float = 0.0

    energy_metrics: Optional[EnergyMetrics] = None
    total_power_watts: float = 0.0
    energy_efficiency: float = 0.0

    label: int = 0
    drift_type: str = "normal"

    def to_dict(self) -> dict:
        result = {
            'timestamp': self.timestamp,
            'total_throughput_mbps': self.total_throughput_mbps,
            'avg_delay_ms': self.avg_delay_ms,
            'avg_loss_rate': self.avg_loss_rate,
            'total_power_watts': self.total_power_watts,
            'energy_efficiency': self.energy_efficiency,
            'label': self.label,
            'drift_type': self.drift_type,
            'links': {k: v.to_dict() for k, v in self.link_metrics.items()},
            'switches': {k: v.to_dict() for k, v in self.switch_metrics.items()},
            'paths': {k: v.to_dict() for k, v in self.path_metrics.items()}
        }
        if self.energy_metrics:
            result['energy'] = self.energy_metrics.to_dict()
        return result


class MetricsCollector:

    def __init__(self,
                 network=None,
                 controller_url: str = "http://127.0.0.1:8080",
                 energy_model: Optional[NetworkEnergyModel] = None):
        self.network = network
        self.controller_url = controller_url
        self.energy_model = energy_model or create_default_energy_model()

        self.is_collecting = False
        self.collect_thread: Optional[threading.Thread] = None
        self.collected_samples: List[NetworkSnapshot] = []

        self._prev_stats: Dict[str, dict] = {}
        self._prev_timestamp: float = 0.0

        self._delay_history: Dict[str, List[float]] = defaultdict(list)

        self.configured_routes: Dict[Tuple[str, str], List[str]] = {}
        self.topology_graph: Optional[nx.Graph] = None
        self.host_to_switch: Dict[str, str] = {}

        # Mapping of node names to Mininet node objects, used for executing commands within the namespace
        self._node_map: Dict[str, Any] = {}

    def set_network(self, network):
        """Set the Mininet network"""
        self.network = network

        if network:
            self._build_topology_graph(network)

            # Establish a mapping from node names to objects
            self._node_map = {}
            for s in network.switches:
                self._node_map[s.name] = s
            for h in network.hosts:
                self._node_map[h.name] = h

            switches = [{'id': s.name, 'num_ports': len(s.intfList())}
                        for s in network.switches]
            links = []
            for link in network.links:
                src = link.intf1.node.name
                dst = link.intf2.node.name
                links.append({
                    'id': f"{src}-{dst}",
                    'src': src,
                    'dst': dst,
                    'capacity': 100
                })
            self.energy_model.initialize_topology(switches, links)

    def _build_topology_graph(self, network):
        self.topology_graph = nx.Graph()
        for switch in network.switches:
            self.topology_graph.add_node(switch.name, type='switch')
        for host in network.hosts:
            self.topology_graph.add_node(host.name, type='host')
            for intf in host.intfList():
                if intf.link:
                    link = intf.link
                    peer_intf = link.intf2 if link.intf1.node == host else link.intf1
                    peer_node = peer_intf.node
                    if hasattr(peer_node, 'dpid'):
                        self.host_to_switch[host.name] = peer_node.name
                        self.topology_graph.add_edge(host.name, peer_node.name)
        for link in network.links:
            src = link.intf1.node.name
            dst = link.intf2.node.name
            self.topology_graph.add_edge(src, dst)

    def set_configured_routes(self, routes: Dict):
        self.configured_routes = routes

    def _get_path_nodes(self, src_host: str, dst_host: str) -> List[str]:
        if (src_host, dst_host) in self.configured_routes:
            return self.configured_routes[(src_host, dst_host)]
        if self.topology_graph and self.topology_graph.has_node(src_host) and self.topology_graph.has_node(dst_host):
            try:
                path = nx.shortest_path(self.topology_graph, src_host, dst_host)
                return [n for n in path if self.topology_graph.nodes[n].get('type') == 'switch']
            except nx.NetworkXNoPath:
                pass
        src_switch = self.host_to_switch.get(src_host)
        dst_switch = self.host_to_switch.get(dst_host)
        if src_switch and dst_switch:
            return [src_switch] if src_switch == dst_switch else [src_switch, dst_switch]
        return []

    def _get_interface_stats(self, intf_name: str, node_name: str) -> dict:

        stats = {}
        node = self._node_map.get(node_name)
        if node is None:
            return {m: 0 for m in ['tx_bytes', 'rx_bytes', 'tx_packets',
                                    'rx_packets', 'tx_errors', 'rx_errors']}
        for metric in ['tx_bytes', 'rx_bytes', 'tx_packets', 'rx_packets',
                       'tx_errors', 'rx_errors']:
            try:
                val = node.cmd(
                    f'cat /sys/class/net/{intf_name}/statistics/{metric} 2>/dev/null'
                ).strip()
                stats[metric] = int(val) if val.isdigit() else 0
            except Exception:
                stats[metric] = 0
        return stats

    def _collect_link_metrics(self, snapshot: NetworkSnapshot):

        if not self.network:
            return

        for link in self.network.links:
            src_node = link.intf1.node
            dst_node = link.intf2.node
            src = src_node.name
            dst = dst_node.name
            link_id = f"{src}-{dst}"

            metrics = LinkMetrics(
                link_id=link_id, src=src, dst=dst,
                timestamp=snapshot.timestamp
            )

            
            intf1_stats = self._get_interface_stats(link.intf1.name, src)
            intf2_stats = self._get_interface_stats(link.intf2.name, dst)

            metrics.tx_bytes = intf1_stats.get('tx_bytes', 0)
            metrics.rx_bytes = intf2_stats.get('rx_bytes', 0)

            
            prev_key_tx = f"{link_id}_tx"
            prev_key_rx = f"{link_id}_rx"
            if self._prev_timestamp > 0:
                dt = snapshot.timestamp - self._prev_timestamp
                if dt > 0:
                    throughput_tx = 0.0
                    throughput_rx = 0.0
                    if prev_key_tx in self._prev_stats:
                        diff = metrics.tx_bytes - self._prev_stats[prev_key_tx]
                        throughput_tx = max(0.0, (diff * 8) / (dt * 1e6))
                    if prev_key_rx in self._prev_stats:
                        diff = metrics.rx_bytes - self._prev_stats[prev_key_rx]
                        throughput_rx = max(0.0, (diff * 8) / (dt * 1e6))
                    metrics.throughput_mbps = max(throughput_tx, throughput_rx)

            self._prev_stats[prev_key_tx] = metrics.tx_bytes
            self._prev_stats[prev_key_rx] = metrics.rx_bytes


            if hasattr(src_node, 'dpid') and hasattr(dst_node, 'dpid'):

                try:
                    tc_out = src_node.cmd(
                        f'tc qdisc show dev {link.intf1.name} 2>/dev/null'
                    )
                    delay_match = re.search(r'delay\s+([\d.]+)ms', tc_out)
                    if delay_match:
                        metrics.delay_ms = float(delay_match.group(1))
                    else:
                        metrics.delay_ms = 0.0
                except Exception:
                    metrics.delay_ms = 0.0
            else:
                metrics.delay_ms = 0.0

            metrics.jitter_ms = 0.0

            # Update the delay history and percentiles
            self._delay_history[link_id].append(metrics.delay_ms)
            if len(self._delay_history[link_id]) > 100:
                self._delay_history[link_id] = self._delay_history[link_id][-100:]
            delays = self._delay_history[link_id]
            if len(delays) >= 10:
                metrics.delay_p10 = np.percentile(delays, 10)
                metrics.delay_p20 = np.percentile(delays, 20)
                metrics.delay_p50 = np.percentile(delays, 50)
                metrics.delay_p80 = np.percentile(delays, 80)
                metrics.delay_p90 = np.percentile(delays, 90)
                metrics.delay_p99 = np.percentile(delays, 99)

            snapshot.link_metrics[link_id] = metrics

    def _collect_switch_metrics(self, snapshot: NetworkSnapshot):
        if not self.network:
            return

        for switch in self.network.switches:
            switch_id = switch.name
            metrics = SwitchMetrics(switch_id=switch_id, timestamp=snapshot.timestamp)

            active_ports = 0
            total_throughput = 0.0

            for intf in switch.intfList():
                if intf.name == 'lo' or intf.name.startswith('lo'):
                    continue
                try:
                    # 修复：传入节点名
                    stats = self._get_interface_stats(intf.name, switch_id)
                    if stats.get('tx_bytes', 0) > 0 or stats.get('rx_bytes', 0) > 0:
                        active_ports += 1

                    port_id = f"{switch_id}_{intf.name}"
                    prev_stats = self._prev_stats.get(port_id, {})
                    if prev_stats and self._prev_timestamp > 0:
                        dt = snapshot.timestamp - self._prev_timestamp
                        if dt > 0:
                            tx_diff = stats.get('tx_bytes', 0) - prev_stats.get('tx_bytes', 0)
                            rx_diff = stats.get('rx_bytes', 0) - prev_stats.get('rx_bytes', 0)
                            port_throughput = max(tx_diff, rx_diff) * 8 / (dt * 1e6)
                            total_throughput += max(0.0, port_throughput)

                    self._prev_stats[port_id] = {
                        'tx_bytes': stats.get('tx_bytes', 0),
                        'rx_bytes': stats.get('rx_bytes', 0)
                    }
                    metrics.port_stats[intf.name] = stats

                except Exception:
                    pass

            metrics.num_active_ports = active_ports
            metrics.total_throughput_mbps = total_throughput
            snapshot.switch_metrics[switch_id] = metrics

    def _collect_path_metrics(self, snapshot: NetworkSnapshot):
        if not self.network:
            return
        hosts = self.network.hosts
        if len(hosts) < 2:
            return

        test_pairs = [(hosts[0], hosts[-1])]
        for (src_name, dst_name) in self.configured_routes.keys():
            src_host = self.network.get(src_name)
            dst_host = self.network.get(dst_name)
            if src_host and dst_host and (src_host, dst_host) not in test_pairs:
                test_pairs.append((src_host, dst_host))

        for src_host, dst_host in test_pairs:
            path_id = f"{src_host.name}-{dst_host.name}"
            path_nodes = self._get_path_nodes(src_host.name, dst_host.name)

            metrics = PathMetrics(
                path_id=path_id,
                src_host=src_host.name,
                dst_host=dst_host.name,
                path_nodes=path_nodes,
                num_hops=len(path_nodes),
                timestamp=snapshot.timestamp
            )

            try:
                cmd = f"ping -c 2 -i 0.2 -W 1 -Q 0xb8 {dst_host.IP()}"
                p = src_host.popen(cmd.split())
                out, err = p.communicate(timeout=3.0)
                result = out.decode('utf-8', errors='ignore')

                loss_match = re.search(r'(\d+)% packet loss', result)
                if loss_match:
                    metrics.e2e_loss_rate = float(loss_match.group(1)) / 100.0
                else:
                    metrics.e2e_loss_rate = 1.0

                delay_match = re.search(r'= [\d\.]+/([\d\.]+)/[\d\.]+/', result)
                if delay_match:
                    metrics.e2e_delay_ms = float(delay_match.group(1))
            except subprocess.TimeoutExpired:
                p.kill()
                metrics.e2e_loss_rate = 1.0
            except Exception:
                pass

            metrics.e2e_throughput_mbps = self._calculate_path_throughput(path_nodes, snapshot)
            metrics.path_power_watts = self._calculate_path_power(path_nodes, snapshot)
            snapshot.path_metrics[path_id] = metrics

    def _calculate_path_power(self, path_nodes, snapshot):
        if not path_nodes:
            return 0.0
        total_power = 0.0
        for node in path_nodes:
            if node in snapshot.switch_metrics:
                total_power += snapshot.switch_metrics[node].power_watts
        for i in range(len(path_nodes) - 1):
            link_id = f"{path_nodes[i]}-{path_nodes[i+1]}"
            alt_link_id = f"{path_nodes[i+1]}-{path_nodes[i]}"
            if link_id in snapshot.link_metrics:
                total_power += snapshot.link_metrics[link_id].power_watts
            elif alt_link_id in snapshot.link_metrics:
                total_power += snapshot.link_metrics[alt_link_id].power_watts
        return total_power

    def _calculate_path_throughput(self, path_nodes, snapshot):
        if not path_nodes:
            return 0.0
        throughputs = []
        for i in range(len(path_nodes) - 1):
            link_id = f"{path_nodes[i]}-{path_nodes[i+1]}"
            alt_link_id = f"{path_nodes[i+1]}-{path_nodes[i]}"
            if link_id in snapshot.link_metrics:
                throughputs.append(snapshot.link_metrics[link_id].throughput_mbps)
            elif alt_link_id in snapshot.link_metrics:
                throughputs.append(snapshot.link_metrics[alt_link_id].throughput_mbps)
        if throughputs:
            return min(throughputs) if min(throughputs) > 0 else max(throughputs)
        return 0.0

    def _calculate_energy_metrics(self, snapshot: NetworkSnapshot):
        switch_utils = {}
        switch_throughputs = {}
        link_utils = {}
        link_rates = {}

        for switch_id, metrics in snapshot.switch_metrics.items():
            port_utils = {}
            for i, (port_name, stats) in enumerate(metrics.port_stats.items()):
                util = min(1.0, metrics.total_throughput_mbps / (100.0 * max(1, metrics.num_active_ports)))
                port_utils[i] = util
            switch_utils[switch_id] = port_utils
            switch_throughputs[switch_id] = metrics.total_throughput_mbps

        for link_id, metrics in snapshot.link_metrics.items():
            link_utils[link_id] = metrics.utilization
            link_rates[link_id] = metrics.throughput_mbps

        energy_metrics = self.energy_model.calculate_network_power(
            switch_utils, switch_throughputs,
            link_utils, link_rates,
            timestamp=snapshot.timestamp
        )

        snapshot.energy_metrics = energy_metrics
        snapshot.total_power_watts = energy_metrics.total_network_power
        snapshot.energy_efficiency = energy_metrics.energy_efficiency

        for switch_id, power in energy_metrics.switch_power.items():
            if switch_id in snapshot.switch_metrics:
                snapshot.switch_metrics[switch_id].power_watts = power

        for link_id, power in energy_metrics.link_power.items():
            if link_id in snapshot.link_metrics:
                snapshot.link_metrics[link_id].power_watts = power

    def _aggregate_network_metrics(self, snapshot: NetworkSnapshot):
        delays = [m.delay_ms for m in snapshot.link_metrics.values() if m.delay_ms > 0]
        if delays:
            snapshot.avg_delay_ms = np.mean(delays)

        snapshot.total_throughput_mbps = sum(
            m.total_throughput_mbps for m in snapshot.switch_metrics.values()
        )

        loss_rates = [m.loss_rate for m in snapshot.link_metrics.values()]
        if loss_rates:
            snapshot.avg_loss_rate = np.mean(loss_rates)

    def collect_snapshot(self) -> NetworkSnapshot:
        timestamp = time.time()
        snapshot = NetworkSnapshot(timestamp=timestamp)

        if not self.network:
            print("DEBUG: network is None in collect_snapshot!")
            return snapshot

        self._collect_link_metrics(snapshot)
        self._collect_switch_metrics(snapshot)
        self._collect_path_metrics(snapshot)
        self._calculate_energy_metrics(snapshot)
        self._aggregate_network_metrics(snapshot)

        self._prev_timestamp = snapshot.timestamp
        return snapshot

    def start_collection(self, interval: float = 1.0):
        if self.is_collecting:
            print("DEBUG: Already collecting, skipping start")
            return

        self.is_collecting = True
        self.collected_samples = []

        def collect_loop():
            print(f"DEBUG: Collection thread started")
            loop_count = 0
            while self.is_collecting:
                try:
                    snapshot = self.collect_snapshot()
                    self.collected_samples.append(snapshot)
                    loop_count += 1
                    if loop_count % 5 == 0:
                        print(f"DEBUG: Collected {loop_count} snapshots so far")
                except Exception as e:
                    print(f"Collection error: {e}")
                    import traceback
                    traceback.print_exc()
                time.sleep(interval)
            print(f"DEBUG: Collection thread ending, total {loop_count} snapshots")

        self.collect_thread = threading.Thread(target=collect_loop, daemon=True)
        self.collect_thread.start()
        print(f"Started metrics collection with interval {interval}s")

    def stop_collection(self) -> List[NetworkSnapshot]:
        self.is_collecting = False
        if self.collect_thread:
            self.collect_thread.join(timeout=5.0)
        return self.collected_samples

    def set_label(self, label: int, drift_type: str = "normal"):
        for sample in self.collected_samples:
            sample.label = label
            sample.drift_type = drift_type

    def export_samples(self, filepath: str):
        data = [s.to_dict() for s in self.collected_samples]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Exported {len(data)} samples to {filepath}")


class EnergyAwareCollector(MetricsCollector):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy_drift_detector = EnergyDriftDetector(self.energy_model)
        self.baseline_snapshot: Optional[NetworkSnapshot] = None

    def set_baseline(self, snapshot: NetworkSnapshot):
        self.baseline_snapshot = snapshot
        if snapshot.energy_metrics:
            self.energy_drift_detector.set_baseline(snapshot.total_power_watts)

    def detect_energy_drift(self, snapshot, intent_max_power, performance_satisfied=True):
        if not snapshot.energy_metrics:
            return {'has_drift': False}
        return self.energy_drift_detector.detect_energy_drift(
            snapshot.energy_metrics, intent_max_power, performance_satisfied
        )

    def localize_energy_anomaly(self, snapshot):
        if not snapshot.energy_metrics:
            return []
        baseline_metrics = (self.baseline_snapshot.energy_metrics
                            if self.baseline_snapshot else None)
        return self.energy_drift_detector.localize_energy_anomaly(
            snapshot.energy_metrics, baseline_metrics
        )


if __name__ == '__main__':
    collector = MetricsCollector()
    print("Testing MetricsCollector...")
    snapshot = collector.collect_snapshot()
    print(f"Snapshot timestamp: {snapshot.timestamp}")
    print(f"Total power: {snapshot.total_power_watts:.2f} W")
