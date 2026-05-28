# utils/energy_model.py
"""
网络能耗模型 (Network Energy Model)

基于学术界常用的线性功耗模型，支持：
- 节点（交换机）功耗估算
- 链路功耗估算
- 路径功耗估算
- 全网功耗统计

参考文献：
- GreenTE: Power-aware traffic engineering (IMC 2010)
- CARPO: Correlation-aware power optimization (INFOCOM 2012)
- Energy-aware routing in SDN (Computer Networks 2018)

功耗模型：
  P_node = P_idle + P_port * num_active_ports + E_dynamic * utilization
  P_link = P_transceiver * 2 + E_per_bit * traffic_rate

核心思想：
  - 性能达标但能耗超标 = 能耗意图漂移（传统监控盲区）
  - 流量聚合可以让空闲链路休眠，降低能耗
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class DeviceState(Enum):
    """设备状态"""
    ACTIVE = "active"       # 活跃状态
    IDLE = "idle"           # 空闲状态
    SLEEP = "sleep"         # 休眠状态（低功耗）


@dataclass
class SwitchEnergyProfile:
    """交换机能耗配置"""
    # 基础功耗参数（单位：瓦特）
    P_chassis: float = 100.0      # 机箱基础功耗
    P_idle_per_port: float = 2.0  # 每端口空闲功耗
    P_active_per_port: float = 5.0  # 每端口活跃功耗
    E_per_gbps: float = 10.0      # 每Gbps流量的额外功耗
    
    # 休眠模式参数
    P_sleep: float = 20.0         # 休眠模式功耗
    sleep_threshold: float = 0.01  # 利用率低于此值可进入休眠
    wakeup_delay_ms: float = 50.0  # 唤醒延迟（毫秒）


@dataclass
class LinkEnergyProfile:
    """链路能耗配置"""
    # 收发器功耗（两端各一个）
    P_transceiver: float = 1.0    # 单个收发器功耗
    E_per_gbps: float = 0.5       # 每Gbps传输功耗
    
    # 链路休眠参数
    P_sleep: float = 0.2          # 休眠模式功耗
    can_sleep: bool = True        # 是否支持休眠


@dataclass 
class EnergyMetrics:
    """能耗指标"""
    timestamp: float = 0.0
    
    # 节点能耗
    switch_power: Dict[str, float] = field(default_factory=dict)
    switch_state: Dict[str, DeviceState] = field(default_factory=dict)
    
    # 链路能耗
    link_power: Dict[str, float] = field(default_factory=dict)
    link_state: Dict[str, DeviceState] = field(default_factory=dict)
    
    # 汇总指标
    total_switch_power: float = 0.0
    total_link_power: float = 0.0
    total_network_power: float = 0.0
    
    # 效率指标
    active_switches: int = 0
    sleeping_switches: int = 0
    active_links: int = 0
    sleeping_links: int = 0
    
    # 能效比（每瓦特传输的Mbps）
    energy_efficiency: float = 0.0  # Mbps/Watt
    
    def to_dict(self) -> dict:
        """转换为字典"""
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
    """
    网络能耗模型
    
    核心功能：
    1. 基于利用率估算设备功耗
    2. 计算路径能耗
    3. 评估流量聚合的节能潜力
    4. 检测能耗漂移
    """
    
    def __init__(self, 
                 switch_profile: Optional[SwitchEnergyProfile] = None,
                 link_profile: Optional[LinkEnergyProfile] = None):
        """
        初始化能耗模型
        
        Args:
            switch_profile: 交换机能耗配置
            link_profile: 链路能耗配置
        """
        self.switch_profile = switch_profile or SwitchEnergyProfile()
        self.link_profile = link_profile or LinkEnergyProfile()
        
        # 网络拓扑信息（运行时设置）
        self.switches: Dict[str, dict] = {}  # switch_id -> {num_ports, ...}
        self.links: Dict[str, dict] = {}     # link_id -> {src, dst, capacity, ...}
        
    def set_topology(self, switches: List[dict], links: List[dict]):
        """
        设置网络拓扑
        
        Args:
            switches: 交换机列表 [{'id': 's1', 'num_ports': 24}, ...]
            links: 链路列表 [{'id': 's1-s2', 'src': 's1', 'dst': 's2', 'capacity': 1000}, ...]
        """
        self.switches = {s['id']: s for s in switches}
        self.links = {l.get('id', f"{l['src']}-{l['dst']}"): l for l in links}
    def initialize_topology(self, switches: List[dict], links: List[dict]):
        self.set_topology(switches, links)
        
    def calculate_switch_power(self, 
                               switch_id: str,
                               port_utilizations: Dict[int, float],
                               total_throughput_mbps: float = 0.0) -> Tuple[float, DeviceState]:
        """
        计算单个交换机的功耗
        
        Args:
            switch_id: 交换机ID
            port_utilizations: 各端口利用率 {port_no: utilization}
            total_throughput_mbps: 总吞吐量（Mbps）
            
        Returns:
            (power_watts, state): 功耗和设备状态
        """
        profile = self.switch_profile
        switch_info = self.switches.get(switch_id, {'num_ports': 24})
        num_ports = switch_info.get('num_ports', 24)
        
        # 计算活跃端口数
        active_ports = sum(1 for u in port_utilizations.values() if u > 0.01)
        avg_utilization = np.mean(list(port_utilizations.values())) if port_utilizations else 0.0
        
        # 判断是否可以休眠
        if avg_utilization < profile.sleep_threshold and active_ports == 0:
            return profile.P_sleep, DeviceState.SLEEP
        
        # 计算功耗
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
        """
        计算单条链路的功耗
        
        Args:
            link_id: 链路ID
            utilization: 链路利用率
            traffic_rate_mbps: 流量速率（Mbps）
            
        Returns:
            (power_watts, state): 功耗和设备状态
        """
        profile = self.link_profile
        
        # 判断是否可以休眠
        if profile.can_sleep and utilization < 0.01:
            return profile.P_sleep, DeviceState.SLEEP
        
        # 计算功耗：两端收发器 + 传输功耗
        power = (2 * profile.P_transceiver + 
                 profile.E_per_gbps * (traffic_rate_mbps / 1000.0))
        
        state = DeviceState.ACTIVE if utilization > 0.01 else DeviceState.IDLE
        
        return power, state
    
    def calculate_path_power(self,
                            path: List[str],
                            traffic_rate_mbps: float) -> float:
        """
        计算路径的能耗
        
        Args:
            path: 路径上的节点列表 ['s1', 's2', 's3']
            traffic_rate_mbps: 流量速率（Mbps）
            
        Returns:
            path_power: 路径总能耗（仅计算边际增量）
        """
        if len(path) < 2:
            return 0.0
        
        total_power = 0.0
        
        # 计算路径上每跳的边际能耗
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            
            # 链路能耗（边际增量）
            link_power = (2 * self.link_profile.P_transceiver +
                         self.link_profile.E_per_gbps * (traffic_rate_mbps / 1000.0))
            total_power += link_power
            
            # 节点转发能耗（边际增量）
            node_power = self.switch_profile.E_per_gbps * (traffic_rate_mbps / 1000.0)
            total_power += node_power
        
        return total_power
    
    def calculate_network_power(self,
                               switch_utils: Dict[str, Dict[int, float]],
                               switch_throughputs: Dict[str, float],
                               link_utils: Dict[str, float],
                               link_rates: Dict[str, float],
                               timestamp: float = 0.0) -> EnergyMetrics:
        """
        计算全网能耗
        
        Args:
            switch_utils: 各交换机端口利用率 {switch_id: {port: util}}
            switch_throughputs: 各交换机吞吐量 {switch_id: throughput_mbps}
            link_utils: 链路利用率 {link_id: utilization}
            link_rates: 链路流量速率 {link_id: rate_mbps}
            timestamp: 时间戳
            
        Returns:
            EnergyMetrics: 能耗指标
        """
        metrics = EnergyMetrics(timestamp=timestamp)
        
        total_throughput = 0.0
        
        # 计算交换机能耗
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
        
        # 计算链路能耗
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
        
        # 计算总功耗
        metrics.total_network_power = metrics.total_switch_power + metrics.total_link_power
        
        # 计算能效比
        if metrics.total_network_power > 0:
            metrics.energy_efficiency = total_throughput / metrics.total_network_power
        
        return metrics
    
    def estimate_optimal_power(self,
                              traffic_matrix: np.ndarray,
                              node_ids: List[str]) -> float:
        """
        估算最优能耗（理想流量聚合情况）
        
        通过将流量聚合到最少的链路上，让其他链路休眠
        
        Args:
            traffic_matrix: 流量矩阵 [N x N]
            node_ids: 节点ID列表
            
        Returns:
            optimal_power: 最优能耗估计
        """
        total_traffic = np.sum(traffic_matrix)
        n_nodes = len(node_ids)
        
        # 假设最优情况：使用最短路径树，其他链路休眠
        # 简化估算：活跃节点数 = 有流量的源/目的节点数
        active_nodes = set()
        for i in range(n_nodes):
            for j in range(n_nodes):
                if traffic_matrix[i, j] > 0:
                    active_nodes.add(i)
                    active_nodes.add(j)
        
        n_active = len(active_nodes)
        n_sleeping = n_nodes - n_active
        
        # 最优功耗估算
        switch_power = (n_active * self.switch_profile.P_chassis +
                       n_sleeping * self.switch_profile.P_sleep)
        
        # 链路功耗（假设活跃链路数 ≈ 活跃节点数 - 1）
        n_active_links = max(0, n_active - 1)
        n_sleeping_links = len(self.links) - n_active_links
        
        link_power = (n_active_links * 2 * self.link_profile.P_transceiver +
                     n_sleeping_links * self.link_profile.P_sleep)
        
        return switch_power + link_power


class EnergyDriftDetector:
    """
    能耗漂移检测器
    
    检测场景：
    1. 性能达标但能耗超标（隐蔽漂移）
    2. 次优路由导致的能耗浪费
    3. 设备异常导致的能耗激增
    """
    
    def __init__(self, energy_model: NetworkEnergyModel):
        self.energy_model = energy_model
        self.baseline_power: Optional[float] = None
        self.power_history: List[float] = []
        
    def set_baseline(self, baseline_power: float):
        """设置基准能耗"""
        self.baseline_power = baseline_power
        
    def update_history(self, current_power: float):
        """更新历史记录"""
        self.power_history.append(current_power)
        if len(self.power_history) > 100:
            self.power_history.pop(0)
    
    def detect_energy_drift(self,
                           current_metrics: EnergyMetrics,
                           intent_max_power: float,
                           performance_satisfied: bool = True) -> dict:
        """
        检测能耗漂移
        
        Args:
            current_metrics: 当前能耗指标
            intent_max_power: 意图约束的最大功耗
            performance_satisfied: 性能是否满足
            
        Returns:
            drift_result: {
                'has_drift': bool,
                'drift_type': str,
                'severity': float,
                'details': dict
            }
        """
        current_power = current_metrics.total_network_power
        self.update_history(current_power)
        
        result = {
            'has_drift': False,
            'drift_type': None,
            'severity': 0.0,
            'details': {}
        }
        
        # 场景1：性能达标但能耗超标（最隐蔽的漂移）
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
            
        # 场景2：能耗突增（相对于历史基准）
        elif len(self.power_history) >= 10:
            avg_power = np.mean(self.power_history[-10:])
            if current_power > avg_power * 1.5:  # 超过历史均值50%
                result['has_drift'] = True
                result['drift_type'] = 'energy_spike'
                result['severity'] = (current_power - avg_power) / avg_power
                result['details'] = {
                    'description': '能耗突然激增',
                    'current_power': current_power,
                    'historical_avg': avg_power
                }
        
        # 场景3：与基准对比
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
        """
        定位能耗异常的设备
        
        Args:
            current_metrics: 当前能耗指标
            baseline_metrics: 基准能耗指标（可选）
            
        Returns:
            anomalies: 异常设备列表
        """
        anomalies = []
        
        # 检测交换机异常
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
        
        # 检测链路异常
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
        
        # 按异常程度排序
        anomalies.sort(key=lambda x: x['anomaly_ratio'], reverse=True)
        
        return anomalies


# 便捷函数
def create_default_energy_model() -> NetworkEnergyModel:
    """创建默认能耗模型"""
    return NetworkEnergyModel(
        switch_profile=SwitchEnergyProfile(),
        link_profile=LinkEnergyProfile()
    )


def estimate_power_from_utilization(utilization: float, 
                                    num_devices: int = 10,
                                    device_type: str = 'switch') -> float:
    """
    快速估算功耗
    
    Args:
        utilization: 平均利用率 (0-1)
        num_devices: 设备数量
        device_type: 设备类型 ('switch' or 'link')
        
    Returns:
        estimated_power: 估算功耗（瓦特）
    """
    if device_type == 'switch':
        profile = SwitchEnergyProfile()
        per_device = profile.P_chassis + profile.E_per_gbps * utilization * 10  # 假设10Gbps容量
    else:
        profile = LinkEnergyProfile()
        per_device = 2 * profile.P_transceiver + profile.E_per_gbps * utilization * 10
    
    return per_device * num_devices


if __name__ == '__main__':
    # 测试能耗模型
    model = create_default_energy_model()
    
    # 设置拓扑
    switches = [{'id': f's{i}', 'num_ports': 24} for i in range(1, 6)]
    links = [
        {'id': 's1-s2', 'src': 's1', 'dst': 's2', 'capacity': 1000},
        {'id': 's2-s3', 'src': 's2', 'dst': 's3', 'capacity': 1000},
        {'id': 's3-s4', 'src': 's3', 'dst': 's4', 'capacity': 1000},
        {'id': 's4-s5', 'src': 's4', 'dst': 's5', 'capacity': 1000},
        {'id': 's1-s3', 'src': 's1', 'dst': 's3', 'capacity': 1000},
    ]
    model.set_topology(switches, links)
    
    # 模拟利用率
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
    
    # 计算能耗
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
    
    # 测试漂移检测
    detector = EnergyDriftDetector(model)
    drift_result = detector.detect_energy_drift(
        metrics,
        intent_max_power=500.0,  # 意图约束
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
