# utils/energy_model.py


import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class DeviceState(Enum):
    """Device State"""
    ACTIVE = "active"       # Active State
    IDLE = "idle"           # Idle State
    SLEEP = "sleep"         # Sleep State (Low Power)


@dataclass
class SwitchEnergyProfile:
    """Switch Energy Profile"""
    # Base power parameters (Unit: Watt)
    P_chassis: float = 100.0      # Chassis base power
    P_idle_per_port: float = 2.0  # Idle power per port
    P_active_per_port: float = 5.0  # Active power per port
    E_per_gbps: float = 10.0      # Extra power per Gbps traffic
    
    # Sleep mode parameters
    P_sleep: float = 20.0         # Sleep mode power
    sleep_threshold: float = 0.01  # Utilization below this value can enter sleep
    wakeup_delay_ms: float = 50.0  # Wakeup delay (milliseconds)


@dataclass
class LinkEnergyProfile:
    """Link Energy Profile"""
    # Transceiver power (one at each end)
    P_transceiver: float = 1.0    # Power of a single transceiver
    E_per_gbps: float = 0.5       # Power per Gbps transmission
    
    # Link sleep parameters
    P_sleep: float = 0.2          # Sleep mode power
    can_sleep: bool = True        # Whether sleep mode is supported


@dataclass 
class EnergyMetrics:
    """Energy Metrics"""
    timestamp: float = 0.0
    
    # Node energy consumption
    switch_power: Dict[str, float] = field(default_factory=dict)
    switch_state: Dict[str, DeviceState] = field(default_factory=dict)
    
    # Link energy consumption
    link_power: Dict[str, float] = field(default_factory=dict)
    link_state: Dict[str, DeviceState] = field(default_factory=dict)
    
    # Summary metrics
    total_switch_power: float = 0.0
    total_link_power: float = 0.0
    total_network_power: float = 0.0
    
    # Efficiency metrics
    active_switches: int = 0
    sleeping_switches: int = 0
    active_links: int = 0
    sleeping_links: int = 0
    
    # Energy efficiency (Mbps per Watt)
    energy_efficiency: float = 0.0  # Mbps/Watt
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'total_switch_power': self.total_switch_power,
            'total_link_power': self.total_link_power,
            'total_network_power': self.total_network_power,
            'active_switches': self.active_switches,
            'sleeping_switches': self.sleeping_switches,
            'active_links': self.active_links,
            'sleeping_links': self.sleeping_links,
            'energy_efficiency': self.energy_efficiency,
            'switch_power': self.switch_power.copy(),
            'link_power': self.link_power.copy()
        }


class NetworkEnergyModel:

    
    def __init__(self, 
                 switch_profile: Optional[SwitchEnergyProfile] = None,
                 link_profile: Optional[LinkEnergyProfile] = None):

        self.switch_profile = switch_profile or SwitchEnergyProfile()
        self.link_profile = link_profile or LinkEnergyProfile()
        
        # Network Topology Information (Runtime Settings)
        self.switches: Dict[str, dict] = {}  # switch_id -> {num_ports, ...}
        self.links: Dict[str, dict] = {}     # link_id -> {src, dst, capacity, ...}
        
    def set_topology(self, switches: List[dict], links: List[dict]):

        self.switches = {s['id']: s for s in switches}
        self.links = {l.get('id', f"{l['src']}-{l['dst']}"): l for l in links}
    def initialize_topology(self, switches: List[dict], links: List[dict]):
        self.set_topology(switches, links)
        
    def calculate_switch_power(self, 
                               switch_id: str,
                               port_utilizations: Dict[int, float],
                               total_throughput_mbps: float = 0.0) -> Tuple[float, DeviceState]:

        profile = self.switch_profile
        switch_info = self.switches.get(switch_id, {'num_ports': 24})
        num_ports = switch_info.get('num_ports', 24)
        
        # Count the number of active ports
        active_ports = sum(1 for u in port_utilizations.values() if u > 0.01)
        avg_utilization = np.mean(list(port_utilizations.values())) if port_utilizations else 0.0
        
        # Determine whether hibernation is possible
        if avg_utilization < profile.sleep_threshold and active_ports == 0:
            return profile.P_sleep, DeviceState.SLEEP
        
        # Calculate power consumption
        # P = P_chassis + P_idle * (total_ports - active_ports) + P_active * active_ports + E_dynamic * throughput
        idle_ports = num_ports - active_ports
        
        power = (profile.P_chassis + 
                 profile.P_idle_per_port * idle_ports +
                 profile.P_active_per_port * active_ports +
                 profile.E_per_gbps * (total_throughput_mbps / 1000.0))
        
        state = DeviceState.ACTIVE if active_ports > 0 else DeviceState.IDLE
        
        return power, state
    
    def calculate_link_power(self,
                            link_id: str,
                            utilization: float,
                            traffic_rate_mbps: float) -> Tuple[float, DeviceState]:

        profile = self.link_profile
        
        # Determine whether hibernation is possible
        if profile.can_sleep and utilization < 0.01:
            return profile.P_sleep, DeviceState.SLEEP
        
        # Calculate power consumption: transceivers at both ends + transmission power
        power = (2 * profile.P_transceiver + 
                 profile.E_per_gbps * (traffic_rate_mbps / 1000.0))
        
        state = DeviceState.ACTIVE if utilization > 0.01 else DeviceState.IDLE
        
        return power, state
    
    def calculate_path_power(self,
                            path: List[str],
                            traffic_rate_mbps: float) -> float:

        if len(path) < 2:
            return 0.0
        
        total_power = 0.0
        
        # Calculate the marginal energy consumption at each hop along the path
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            
            # Link energy consumption (marginal increase)
            link_power = (2 * self.link_profile.P_transceiver +
                         self.link_profile.E_per_gbps * (traffic_rate_mbps / 1000.0))
            total_power += link_power
            
            # Node forwarding energy consumption (marginal increase)
            node_power = self.switch_profile.E_per_gbps * (traffic_rate_mbps / 1000.0)
            total_power += node_power
        
        return total_power
    
    def calculate_network_power(self,
                               switch_utils: Dict[str, Dict[int, float]],
                               switch_throughputs: Dict[str, float],
                               link_utils: Dict[str, float],
                               link_rates: Dict[str, float],
                               timestamp: float = 0.0) -> EnergyMetrics:

        metrics = EnergyMetrics(timestamp=timestamp)
        
        total_throughput = 0.0
        
        # Calculate switch energy consumption
        for switch_id in self.switches:
            port_utils = switch_utils.get(switch_id, {})
            throughput = switch_throughputs.get(switch_id, 0.0)
            total_throughput += throughput
            
            power, state = self.calculate_switch_power(switch_id, port_utils, throughput)
            
            metrics.switch_power[switch_id] = power
            metrics.switch_state[switch_id] = state
            metrics.total_switch_power += power
            
            if state == DeviceState.SLEEP:
                metrics.sleeping_switches += 1
            else:
                metrics.active_switches += 1
        
        # Calculate link energy consumption
        for link_id in self.links:
            utilization = link_utils.get(link_id, 0.0)
            rate = link_rates.get(link_id, 0.0)
            
            power, state = self.calculate_link_power(link_id, utilization, rate)
            
            metrics.link_power[link_id] = power
            metrics.link_state[link_id] = state
            metrics.total_link_power += power
            
            if state == DeviceState.SLEEP:
                metrics.sleeping_links += 1
            else:
                metrics.active_links += 1
        
        # Calculate total power consumption
        metrics.total_network_power = metrics.total_switch_power + metrics.total_link_power
        
        # Calculate energy efficiency
        if metrics.total_network_power > 0:
            metrics.energy_efficiency = total_throughput / metrics.total_network_power
        
        return metrics
    
    def estimate_optimal_power(self,
                              traffic_matrix: np.ndarray,
                              node_ids: List[str]) -> float:

        total_traffic = np.sum(traffic_matrix)
        n_nodes = len(node_ids)
        
        # Assuming the best-case scenario: using a shortest-path tree, with all other links in sleep mode
        # Simplified estimate: Number of active nodes = Number of source/destination nodes with traffic
        active_nodes = set()
        for i in range(n_nodes):
            for j in range(n_nodes):
                if traffic_matrix[i, j] > 0:
                    active_nodes.add(i)
                    active_nodes.add(j)
        
        n_active = len(active_nodes)
        n_sleeping = n_nodes - n_active
        
        # Optimal Power Consumption Estimate
        switch_power = (n_active * self.switch_profile.P_chassis +
                       n_sleeping * self.switch_profile.P_sleep)
        
        # Link power consumption (assuming the number of active links ≈ the number of active nodes - 1)
        n_active_links = max(0, n_active - 1)
        n_sleeping_links = len(self.links) - n_active_links
        
        link_power = (n_active_links * 2 * self.link_profile.P_transceiver +
                     n_sleeping_links * self.link_profile.P_sleep)
        
        return switch_power + link_power


class EnergyDriftDetector:

    
    def __init__(self, energy_model: NetworkEnergyModel):
        self.energy_model = energy_model
        self.baseline_power: Optional[float] = None
        self.power_history: List[float] = []
        
    def set_baseline(self, baseline_power: float):
        """Set the baseline energy consumption"""
        self.baseline_power = baseline_power
        
    def update_history(self, current_power: float):
        """Update the power consumption history"""
        self.power_history.append(current_power)
        if len(self.power_history) > 100:
            self.power_history.pop(0)
    
    def detect_energy_drift(self,
                           current_metrics: EnergyMetrics,
                           intent_max_power: float,
                           performance_satisfied: bool = True) -> dict:

        current_power = current_metrics.total_network_power
        self.update_history(current_power)
        
        result = {
            'has_drift': False,
            'drift_type': None,
            'severity': 0.0,
            'details': {}
        }
        
        # Scene 1: Performance satisfied but energy consumption exceeds limits (most covert drift)
        if performance_satisfied and current_power > intent_max_power:
            result['has_drift'] = True
            result['drift_type'] = 'hidden_energy_drift'
            result['severity'] = (current_power - intent_max_power) / intent_max_power
            result['details'] = {
                'description': '性能达标但能耗超出意图约束（传统监控盲区）',
                'current_power': current_power,
                'intent_max_power': intent_max_power,
                'excess_power': current_power - intent_max_power
            }
            
        # Scene 2: Sudden surge in energy consumption (relative to historical benchmarks)
        elif len(self.power_history) >= 10:
            avg_power = np.mean(self.power_history[-10:])
            if current_power > avg_power * 1.5:  # 50% above the historical average
                result['has_drift'] = True
                result['drift_type'] = 'energy_spike'
                result['severity'] = (current_power - avg_power) / avg_power
                result['details'] = {
                    'description': '能耗突然激增',
                    'current_power': current_power,
                    'historical_avg': avg_power
                }
        
        # Scene 3: Comparison with the Baseline
        elif self.baseline_power and current_power > self.baseline_power * 1.3:
            result['has_drift'] = True
            result['drift_type'] = 'baseline_exceeded'
            result['severity'] = (current_power - self.baseline_power) / self.baseline_power
            result['details'] = {
                'description': '能耗超过基准值',
                'current_power': current_power,
                'baseline_power': self.baseline_power
            }
        
        return result
    
    def localize_energy_anomaly(self,
                               current_metrics: EnergyMetrics,
                               baseline_metrics: Optional[EnergyMetrics] = None) -> List[dict]:

        anomalies = []
        
        # Detect switch anomalies
        for switch_id, power in current_metrics.switch_power.items():
            baseline_power = (baseline_metrics.switch_power.get(switch_id, power) 
                            if baseline_metrics else power * 0.7)
            
            if power > baseline_power * 1.5:
                anomalies.append({
                    'device_type': 'switch',
                    'device_id': switch_id,
                    'current_power': power,
                    'baseline_power': baseline_power,
                    'anomaly_ratio': power / baseline_power,
                    'state': current_metrics.switch_state.get(switch_id, DeviceState.ACTIVE).value
                })
        
        # Detect link anomalies
        for link_id, power in current_metrics.link_power.items():
            baseline_power = (baseline_metrics.link_power.get(link_id, power)
                            if baseline_metrics else power * 0.7)
            
            if power > baseline_power * 1.5:
                anomalies.append({
                    'device_type': 'link',
                    'device_id': link_id,
                    'current_power': power,
                    'baseline_power': baseline_power,
                    'anomaly_ratio': power / baseline_power,
                    'state': current_metrics.link_state.get(link_id, DeviceState.ACTIVE).value
                })
        
        # Sort by anomaly severity
        anomalies.sort(key=lambda x: x['anomaly_ratio'], reverse=True)
        
        return anomalies


# Utility Functions
def create_default_energy_model() -> NetworkEnergyModel:
    """Create the default energy model"""
    return NetworkEnergyModel(
        switch_profile=SwitchEnergyProfile(),
        link_profile=LinkEnergyProfile()
    )


def estimate_power_from_utilization(utilization: float, 
                                    num_devices: int = 10,
                                    device_type: str = 'switch') -> float:

    if device_type == 'switch':
        profile = SwitchEnergyProfile()
        per_device = profile.P_chassis + profile.E_per_gbps * utilization * 10  # Assuming a capacity of 10 Gbps
    else:
        profile = LinkEnergyProfile()
        per_device = 2 * profile.P_transceiver + profile.E_per_gbps * utilization * 10
    
    return per_device * num_devices


if __name__ == '__main__':
    # Testing the Energy Consumption Model
    model = create_default_energy_model()
    
    # Set Topology
    switches = [{'id': f's{i}', 'num_ports': 24} for i in range(1, 6)]
    links = [
        {'id': 's1-s2', 'src': 's1', 'dst': 's2', 'capacity': 1000},
        {'id': 's2-s3', 'src': 's2', 'dst': 's3', 'capacity': 1000},
        {'id': 's3-s4', 'src': 's3', 'dst': 's4', 'capacity': 1000},
        {'id': 's4-s5', 'src': 's4', 'dst': 's5', 'capacity': 1000},
        {'id': 's1-s3', 'src': 's1', 'dst': 's3', 'capacity': 1000},
    ]
    model.set_topology(switches, links)
    
    # Simulated utilization rate
    switch_utils = {
        's1': {1: 0.5, 2: 0.3},
        's2': {1: 0.4, 2: 0.2},
        's3': {1: 0.6, 2: 0.1},
        's4': {1: 0.2, 2: 0.0},
        's5': {1: 0.1, 2: 0.0},
    }
    switch_throughputs = {'s1': 500, 's2': 400, 's3': 600, 's4': 200, 's5': 100}
    link_utils = {'s1-s2': 0.5, 's2-s3': 0.4, 's3-s4': 0.2, 's4-s5': 0.1, 's1-s3': 0.3}
    link_rates = {'s1-s2': 500, 's2-s3': 400, 's3-s4': 200, 's4-s5': 100, 's1-s3': 300}
    
    # Calculate energy consumption
    metrics = model.calculate_network_power(
        switch_utils, switch_throughputs,
        link_utils, link_rates,
        timestamp=1.0
    )
    
    print("=" * 60)
    print("Network Energy Metrics")
    print("=" * 60)
    print(f"Total Switch Power: {metrics.total_switch_power:.2f} W")
    print(f"Total Link Power: {metrics.total_link_power:.2f} W")
    print(f"Total Network Power: {metrics.total_network_power:.2f} W")
    print(f"Active Switches: {metrics.active_switches}")
    print(f"Sleeping Switches: {metrics.sleeping_switches}")
    print(f"Energy Efficiency: {metrics.energy_efficiency:.2f} Mbps/W")
    
    # Drift Detection Test
    detector = EnergyDriftDetector(model)
    drift_result = detector.detect_energy_drift(
        metrics,
        intent_max_power=500.0,  # Intent Constraints
        performance_satisfied=True
    )
    
    print("\n" + "=" * 60)
    print("Energy Drift Detection")
    print("=" * 60)
    print(f"Has Drift: {drift_result['has_drift']}")
    if drift_result['has_drift']:
        print(f"Drift Type: {drift_result['drift_type']}")
        print(f"Severity: {drift_result['severity']:.2%}")
        print(f"Details: {drift_result['details']}")
