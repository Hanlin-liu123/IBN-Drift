# experiments/generate_real_trace_dataset.py
"""
使用真实流量数据生成数据集（含能耗特征）

核心改动（v2 - 意图驱动版）：
1. 新增 IntentGenerator：为每次实验随机生成多样化的意图约束
2. 每个 sample 附带完整的 intent 字段（性能约束 + 能耗约束 + 路径约束）
3. 漂移标签从意图约束自动推导（Intent.check_all），而非硬编码
4. 同一网络状态在不同意图下可能得到不同的漂移标签

使用方法：
    # 启动Controller（终端1）
    cd ~/intent_drift_platform_new2_realtrace
    source ~/ryu_env_py310/bin/activate
    ryu-manager ryu_controller/intent_controller.py --ofp-tcp-listen-port 6653 --wsapi-port 8080
    
    # 运行数据生成（终端2）
    cd ~/intent_drift_platform_new2_realtrace
    sudo ~/ryu_env_py310/bin/python experiments/generate_real_trace_dataset.py --quick
"""
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

# 真实流量相关
from utils.mawi_parser import MAWIParser, SyntheticMAWIProfile
from utils.sndlib_parser import SNDlibParser
from utils.real_traffic_replay import RealTrafficReplayer, TrafficMatrixScaler
from utils.routing_applier import RoutingApplier, apply_routing_to_controller
# 能耗模型
from utils.energy_model import (
    NetworkEnergyModel, EnergyMetrics, EnergyDriftDetector,
    SwitchEnergyProfile, LinkEnergyProfile, create_default_energy_model
)
# 意图解析器（核心改动：真正使用它）
from utils.intent_parser import (
    Intent, IntentParser, IntentType,
    PerformanceConstraints, EnergyConstraints, PathConstraints,
    Constraint, ConstraintOperator
)


# ============================================================
# 新增：意图生成器
# ============================================================
class IntentGenerator:
    """
    为每次实验生成多样化的意图约束
    
    设计原则：
    - 意图约束不能全部固定，否则无法体现"同一状态在不同意图下漂移类型不同"
    - 约束值在合理范围内随机采样，模拟真实运维中不同业务的差异化需求
    - 每个意图包含三类约束：性能、能耗、路径
    
    约束范围参考：
    - 时延：视频流(20-50ms), Web(50-200ms), 文件传输(100-500ms)
    - 丢包率：实时音视频(0.1%-1%), Web(1%-5%), 批量传输(1%-10%)
    - 功耗阈值：紧约束(500-1000W), 中等(1000-2000W), 松约束(2000-5000W)
    - 能效比：高效(3-5 Mbps/W), 中等(1-3 Mbps/W), 低要求(0.5-1 Mbps/W)
    """
    
    # 业务类型模板
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
            'delay_range': (100, 300),       # 放宽性能
            'loss_range': (0.02, 0.08),
            'bandwidth_range': (1, 10),
            'jitter_range': (20, 50),
            'power_range': (300, 800),       # 严格能耗
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
        """
        生成一个随机意图
        
        Args:
            template_name: 业务类型模板名，None则随机选择
            src_host: 源主机
            dst_host: 目的主机
            path_nodes: 路径经过的交换机列表（可选）
        
        Returns:
            Intent 对象
        """
        if template_name is None:
            template_name = self.rng.choice(list(self.INTENT_TEMPLATES.keys()))
        
        template = self.INTENT_TEMPLATES[template_name]
        self._intent_counter += 1
        
        # 在模板范围内随机采样约束值
        delay_threshold = float(self.rng.uniform(*template['delay_range']))
        loss_threshold = float(self.rng.uniform(*template['loss_range']))
        bandwidth_threshold = float(self.rng.uniform(*template['bandwidth_range']))
        jitter_threshold = float(self.rng.uniform(*template['jitter_range']))
        power_threshold = float(self.rng.uniform(*template['power_range']))
        efficiency_threshold = float(self.rng.uniform(*template['efficiency_range']))
        max_hops = int(self.rng.randint(*template['max_hops_range']))
        
        # 构建性能约束
        performance = PerformanceConstraints(
            delay=Constraint("end_to_end_delay", ConstraintOperator.LE, delay_threshold, "ms"),
            loss=Constraint("packet_loss_rate", ConstraintOperator.LE, loss_threshold, "ratio"),
            bandwidth=Constraint("throughput", ConstraintOperator.GE, bandwidth_threshold, "Mbps"),
            jitter=Constraint("delay_jitter", ConstraintOperator.LE, jitter_threshold, "ms"),
        )
        
        # 构建能耗约束
        energy = EnergyConstraints(
            max_power=Constraint("total_network_power", ConstraintOperator.LE, power_threshold, "watts"),
            min_efficiency=Constraint("energy_efficiency", ConstraintOperator.GE, efficiency_threshold, "Mbps/W"),
        )
        
        # 构建路径约束
        # 关键：waypoints 不应该是整条路径（那样太刚性），而是从路径的
        # 中间节点随机挑 1-2 个作为"必经节点"（类似服务链中的防火墙/DPI）
        if path_nodes and len(path_nodes) >= 3:
            middle_nodes = path_nodes[1:-1]  # 排除首尾
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
        
        # 构建流量匹配
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
        """生成一批多样化意图（覆盖不同业务类型）"""
        intents = []
        templates = list(self.INTENT_TEMPLATES.keys())
        for i in range(n):
            template = templates[i % len(templates)]
            intent = self.generate_intent(template, src_host, dst_host, path_nodes)
            intents.append(intent)
        return intents
    
    @staticmethod
    def intent_to_dict(intent: Intent) -> dict:
        """
        将 Intent 对象转为可序列化的字典
        
        这个字典会作为 sample['intent'] 存入数据集，
        后续 Intent Encoder 从这里读取约束向量
        """
        result = {
            'intent_id': intent.intent_id,
            'name': intent.name,
            'type': intent.intent_type.value,
            'priority': intent.priority,
            'match': intent.match,
        }
        
        # 性能约束 → 数值化
        if intent.performance:
            perf = intent.performance
            result['performance_constraints'] = {
                'delay_threshold_ms': perf.delay.threshold if perf.delay else None,
                'loss_threshold': perf.loss.threshold if perf.loss else None,
                'bandwidth_threshold_mbps': perf.bandwidth.threshold if perf.bandwidth else None,
                'jitter_threshold_ms': perf.jitter.threshold if perf.jitter else None,
            }
        
        # 能耗约束 → 数值化
        if intent.energy:
            eng = intent.energy
            result['energy_constraints'] = {
                'max_power_watts': eng.max_power.threshold if eng.max_power else None,
                'min_efficiency_mbps_per_w': eng.min_efficiency.threshold if eng.min_efficiency else None,
            }
        
        # 路径约束 → 数值化
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
        """
        根据意图约束自动推导 clause-level 多标签漂移
        
        论文定位：clause-level semantic drift prediction
        
        不再使用单一的 drift_label (0-3)，而是输出三个独立的 clause 违约标志：
        - perf clause: performance constraints (delay/loss/bandwidth/jitter)
        - path clause: path semantic constraints (waypoints/avoid_nodes/max_hops)
        - energy clause: energy constraints (max_power/min_efficiency)
        
        三个 clause 可以同时违约，支持混合漂移的真实建模。
        
        Returns:
            {
                'clause_labels': {
                    'perf': 0/1,
                    'path': 0/1,
                    'energy': 0/1,
                },
                'has_any_drift': bool,
                'performance_satisfied': bool,
                'energy_satisfied': bool,
                'path_satisfied': bool,
                'violations': list,
                'path_violations': list,    # 细化的路径违约原因
                'auto_labeled': True,
                'metrics_used': dict,
                # 以下为向后兼容字段
                'drift_label': int,   # 单一标签（按优先级聚合，仅用于兼容旧代码）
                'drift_type': str,
            }
        """
        # ============================================================
        # 1. 构造性能指标（只提取意图匹配流的路径）
        # ============================================================
        metrics = {
            'delay_ms': snapshot_dict.get('avg_delay_ms', 0),
            'loss_rate': snapshot_dict.get('avg_loss_rate', 0),
            'total_power_watts': snapshot_dict.get('total_power_watts', 0),
            'energy_efficiency': snapshot_dict.get('energy_efficiency', 0),
        }
        
        paths = snapshot_dict.get('paths', {})
        
        # 提取意图的 src/dst，只对匹配流做 perf 检查
        intent_match = getattr(intent, 'match', None) or {}
        if not isinstance(intent_match, dict):
            intent_match = {}
        intent_src = intent_match.get('src', '')
        intent_dst = intent_match.get('dst', '')
        
        if paths:
            # 优先找意图匹配的流路径
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
                # 只用意图匹配流的 e2e 指标
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
                # 没找到匹配流，fallback 到全网平均（保持原逻辑）
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
        # 2. Performance / Energy clause 检查
        # ============================================================
        # 修复：对无流量场景的 loss 保护
        # Mininet 中很多流正常状态下吞吐为 0，此时 loss_rate=1.0 是正常的
        # 不应因此触发 perf clause 违约
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
        # 3. Path clause 检查（语义约束 + 路径可用性）
        # 包括: max_hops, waypoints (必经), avoid_nodes (禁止),
        #       以及"路径事实上不可用"（端到端 delay/loss 异常）
        # 重要：只对"匹配意图流"的路径做检查，而不是对全网所有路径
        # ============================================================
        path_ok = True
        path_violations = []
        derived_drift_location = []  # 从观测到的异常路径推导出的可疑链路列表
        
        if intent.path:
            waypoints = intent.path.waypoints if intent.path.waypoints else []
            avoid_nodes = intent.path.avoid_nodes if intent.path.avoid_nodes else []
            max_hops = intent.path.max_hops
            
            # 提取意图匹配的流
            intent_match = getattr(intent, 'match', None) or {}
            if not isinstance(intent_match, dict):
                intent_match = {}
            intent_src = intent_match.get('src', '')
            intent_dst = intent_match.get('dst', '')
            
            # 提取意图的 delay 阈值，用于"路径不可用"检测
            intent_delay_threshold = None
            if intent.performance and intent.performance.delay:
                intent_delay_threshold = intent.performance.delay.threshold
            
            for path_id, p_data in paths.items():
                src_host = p_data.get('src_host', '')
                dst_host = p_data.get('dst_host', '')
                
                # 只对匹配意图的流路径做语义检查
                # 如果意图没有指定 src/dst，兜底对所有路径做检查
                if intent_src and intent_dst:
                    path_matches_intent = (
                        (src_host == intent_src and dst_host == intent_dst) or
                        (src_host == intent_dst and dst_host == intent_src)
                    )
                    if not path_matches_intent:
                        continue
                
                path_nodes = p_data.get('path_nodes', [])
                num_hops = p_data.get('num_hops', 0)
                
                # 3.1 hop 数约束
                if max_hops > 0 and num_hops > max_hops:
                    path_ok = False
                    path_violations.append(
                        f"hops[{path_id}]: {num_hops} > {max_hops}"
                    )
                
                # 3.2 waypoints 约束（必经节点）
                if waypoints and path_nodes:
                    missing = [wp for wp in waypoints if wp not in path_nodes]
                    if missing:
                        path_ok = False
                        path_violations.append(
                            f"waypoints[{path_id}]: missing {missing}"
                        )
                
                # 3.3 avoid_nodes 约束（禁止节点）
                if avoid_nodes and path_nodes:
                    forbidden = [an for an in avoid_nodes if an in path_nodes]
                    if forbidden:
                        path_ok = False
                        path_violations.append(
                            f"avoid_nodes[{path_id}]: forbidden {forbidden}"
                        )
                
        
        # 注意：这里不再根据注入类型强制覆盖 path 标签。
        # path clause 必须完全由 snapshot 中观测到的路径语义推导，
        # 避免标签混入生成器先验。
        
        # ============================================================
        # 4. Clause-level 多标签输出
        # ============================================================
        clause_labels = {
            'perf': int(not perf_ok),
            'path': int(not path_ok),
            'energy': int(not energy_ok),
        }
        has_any_drift = any(clause_labels.values())
        
        # ============================================================
        # 5. 向后兼容：推导单一 drift_label（按优先级 path > perf > energy）
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
    """
    使用真实流量数据生成数据集
    
    支持四种漂移类型：
    - Label 0: 正常 (normal)
    - Label 1: 性能漂移 (performance_drift) - 时延/丢包超标
    - Label 2: 路径漂移 (path_drift) - 路由改变
    - Label 3: 能耗漂移 (energy_drift) - 性能达标但能耗超标（隐蔽漂移）
    
    v2 改动：
    - 漂移标签从意图约束自动推导，不再硬编码
    - 每个 sample 包含完整的 intent 字段
    """
    
    # 配置常量（参照BNN-UPC）
    TOPOLOGIES = {
        'train': ['geant'],
        'test': ['abilene', 'germany50', 'nobel-germany']
    }
    
    NUM_ROUTING_VARIANTS = 26  # 每个拓扑26种路由变体
    
    SCHEDULING_POLICIES = ['FIFO', 'SP', 'WFQ', 'DRR']
    NUM_SCHEDULING_CONFIGS = 100
    
    QUEUE_SIZES = [8000, 16000, 32000, 64000]  # bits
    
    # 漂移类型配置
    DRIFT_TYPES = {
        'normal': 0,
        'performance': 1,
        'path': 2,
        'energy': 3  # 新增：能耗漂移
    }
    
    def __init__(self, config):
        self.config = config
        self.output_dir = config.get('output_dir', 'data/real_trace_dataset')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化解析器
        self.sndlib_parser = SNDlibParser(config.get('sndlib_dir', 'data/real_traces/sndlib'))
        self.traffic_profile = None
        
        # 初始化能耗模型
        self.energy_model = create_default_energy_model()
        
        # 配置能耗参数
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
        
        # ============================================================
        # 核心改动：初始化意图生成器（替代硬编码的 intent_max_power）
        # ============================================================
        self.intent_generator = IntentGenerator(
            seed=config.get('intent_seed', 42)
        )
        # 每次实验生成多少个不同的意图用于标注
        self.num_intents_per_experiment = config.get('num_intents_per_experiment', 3)
        
        # 保留原有的 intent_max_power 作为兜底默认值
        self.intent_max_power = config.get('intent_max_power', 1500.0)
        
        # 漂移分布配置
        self.drift_distribution = config.get('drift_distribution', {
            'normal': 0.4,
            'performance': 0.2,
            'path': 0.2,
            'energy': 0.2
        })
        # 路由应用器
        self.routing_applier = RoutingApplier(
            config.get('controller_url', 'http://127.0.0.1:8080')
        )

    def prepare_traffic_profile(self):
        """准备流量配置文件"""
        profile_path = self.config.get('traffic_profile_path')
    
        # 检查是否有真实统计特征文件
        mawi_stats_path = self.config.get('mawi_stats_path', 'profiles/mawi_real_stats.json')
    
        if profile_path and os.path.exists(profile_path):
            # 使用真实的 MAWI pcap（不推荐，会很慢）
            print("Parsing real MAWI pcap file...")
            parser = MAWIParser(profile_path)
            parser.parse_pcap(max_packets=self.config.get('max_packets', 50000))
            self.traffic_profile = parser.generate_traffic_profile()
        else:
            # 使用统计特征生成合成流量（推荐）
            print("Generating synthetic MAWI profile...")
        
            if os.path.exists(mawi_stats_path):
                # 加载真实统计特征
                synth = SyntheticMAWIProfile(profile_path=mawi_stats_path)
                print(f"  Using real statistics from {mawi_stats_path}")
            else:
                # 使用默认硬编码特征
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
        """准备拓扑配置"""
        topologies = {}
        
        # 只使用配置中指定的拓扑
        topo_list = self.config.get('topologies', self.TOPOLOGIES['train'])
        if isinstance(topo_list, str):
            topo_list = [topo_list]
        
        for topo_name in topo_list:
            print(f"Preparing topology: {topo_name}")
            
            # 尝试从SNDlib加载
            network_data = self.sndlib_parser.parse_network(topo_name)
            
            if network_data:
                # 转换为Mininet配置
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
                
                # 初始化能耗模型的拓扑
                self._init_energy_model_topology(topo_config)
                
            else:
                # 使用预定义配置
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
        """初始化能耗模型的拓扑信息"""
        switches = []
        links = []
        
        # 从拓扑配置中提取交换机和链路
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
        """生成调度策略配置"""
        configs = []
        
        for i in range(num_configs):
            config = {
                'id': i,
                'nodes': {}
            }
            
            # 每个节点随机选择策略
            for node_idx in range(50):  # 假设最多50个节点
                policy = np.random.choice(self.SCHEDULING_POLICIES)
                queue_size = np.random.choice(self.QUEUE_SIZES)
                
                node_config = {
                    'policy': policy,
                    'queue_size': queue_size
                }
                
                # 如果是WFQ或DRR，添加权重
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
    # Baseline 自适应：根据真实仿真状态调整意图阈值
    # 
    # 问题背景：意图模板里的硬编码阈值（如 efficiency ≥ 0.5 Mbps/W）
    # 远远超出 Mininet 仿真的真实状态（实际 efficiency ≈ 0.006 Mbps/W），
    # 导致 normal 段的所有快照都被错误标记为 energy clause 违约。
    # 
    # 解决方案：在每次实验开始时先采集一段 baseline normal 快照，
    # 用 baseline 的中位数指标动态生成意图阈值，让 normal 段大概率满足约束。
    # ============================================================
    def _collect_baseline_metrics(self, collector, duration: int = 10) -> Dict[str, float]:
        """
        采集 baseline 阶段的真实指标，用于自适应意图阈值
        
        关键修正：
        - throughput 用 PATH-LEVEL e2e_throughput_mbps（单条流的吞吐），
          不能用 total_throughput_mbps（那是全网总和，会导致阈值高出 100x）
        - 取 30th 百分位作为代表，确保大部分流都能达到这个吞吐
        - power 仍然用 global total（power 本来就是全网指标）
        
        Args:
            collector: EnergyAwareCollector
            duration: baseline 采集时长（秒）
        
        Returns:
            dict with 'delay_ms', 'loss_rate', 'throughput_mbps', 
                      'total_power', 'efficiency' (代表性值)
        """
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
        
        # 收集所有 path-level 的指标（每个时刻的每条流是一个数据点）
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
            
            # power 和 efficiency 是全网指标，保留全局
            if d.get('total_power_watts', 0) > 0:
                powers.append(float(d.get('total_power_watts', 0)))
            if d.get('energy_efficiency', 0) > 0:
                effs.append(float(d.get('energy_efficiency', 0)))
        
        # 关键：throughput 取 30th 百分位（让 70% 的流都能达到）
        # 中位数太严格（一半的流达不到），所以用 P30
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
        
        # 额外打印分布信息，方便诊断
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
        """
        基于 baseline 计算自适应的意图模板
        
        修复版：对 power 和 efficiency 阈值加绝对值下界/上界保护，
        避免 baseline 采集时刻偏低导致正常运行时持续超标。
        
        实测正常阶段分布（GÉANT 22-switch 拓扑）：
          delay:       P95=2.5ms,  max=5.5ms
          loss:        P90=1.0 (已在 auto_label 中通过无流量保护处理)
          throughput:  P50=1.3Mbps
          power:       P95=2542W,  max=2545W
          efficiency:  P50=0.37,   min≈0.05
        
        POWER_FLOOR = 2600W：确保 power 阈值 > 正常运行 max (2545W)
        EFFICIENCY_CEIL = 0.04：确保 efficiency 阈值 < 正常运行 min (~0.05)
        """
        b = baseline
        
        # 绝对值保护下界/上界（根据实测数据校准）
        POWER_FLOOR = 2600.0      # 正常运行 max=2545W，加 ~55W 余量
        EFFICIENCY_CEIL = 0.04    # 正常运行 min≈0.05，留余量
        
        # tight_qos: 严格性能要求，能耗宽松
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
        
        # balanced: 平衡型意图
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
        
        # loose_energy: 宽松性能，严格能耗（论文核心故事：节能优先意图）
        loose_energy = {
            'name': '节能优先意图',
            'type': IntentType.ENERGY_SAVING,
            'delay_range':      (max(b['delay_ms'] * 8, 30),  max(b['delay_ms'] * 12, 80)),
            'loss_range':       (max(b['loss_rate'] + 0.03, 0.05),  max(b['loss_rate'] + 0.08, 0.12)),
            'bandwidth_range':  (b['throughput_mbps'] * 0.1,  b['throughput_mbps'] * 0.2),
            'jitter_range':     (max(b['delay_ms'] * 8, 20),  max(b['delay_ms'] * 15, 60)),
            # 节能意图的 power 阈值略紧但仍高于正常运行
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
    
    # ============================================================
    # 核心改动：_label_snapshots 统一标注方法
    # ============================================================
    def _label_snapshots(self, snapshots, intent, injected_drift_type='normal',
                         drift_location=None, drift_params=None,
                         topo_name='', sched_config_id=0,
                         baseline_routing=None):
        """
        用意图约束自动标注一批快照
        
        这是替代原来硬编码 sample['drift_label'] = X 的统一方法。
        
        Args:
            snapshots: NetworkSnapshot 列表
            intent: Intent 对象
            injected_drift_type: 实际注入的漂移类型 ('normal', 'performance', 'path', 'energy')
            drift_location: 漂移注入位置
            drift_params: 漂移注入参数
            topo_name: 拓扑名称
            sched_config_id: 调度配置ID
            baseline_routing: 实验开始时的初始 routing dict（未被drift修改的版本）
                              用于 SAFLA-style baseline 的 I (declared intent set)
        
        Returns:
            sample 字典列表
        """
        samples = []
        intent_dict = IntentGenerator.intent_to_dict(intent)
        
        # 将 baseline_routing 的 tuple key 转为 string (JSON 不支持 tuple key)
        # 格式："src-dst" -> [s1, s5, ...]
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
            # 核心：自动标注（而非硬编码）
            # ============================================================
            label_result = IntentGenerator.auto_label(
                intent=intent,
                snapshot_dict=sample,
                path_changed=(injected_drift_type == 'path'),
                injected_drift_type=injected_drift_type,
            )
            
            # ============================================================
            # 写入 clause-level 多标签（核心字段）
            # ============================================================
            sample['clause_labels'] = label_result['clause_labels']
            sample['has_any_drift'] = label_result['has_any_drift']
            
            # 向后兼容：单一 drift_label
            sample['drift_label'] = label_result['drift_label']
            sample['drift_type'] = label_result['drift_type']
            sample['label'] = label_result['drift_label']
            
            # 写入意图（完整的约束信息，供 Intent Encoder 使用）
            sample['intent'] = intent_dict
            
            # 写入自动标注的详细信息
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
            
            # 写入漂移注入信息（ground truth）
            # 优先使用注入时的 drift_location（真实的注入位置）；
            # 如果没有注入（normal 段的偶发违约），回退用 auto_label 从观测路径推导出的位置
            if drift_location:
                sample['drift_location'] = drift_location
            elif label_result.get('has_any_drift') and label_result.get('derived_drift_location'):
                # normal 段观测到 clause 违约（比如路径延迟飙高），
                # 从 auto_label 里提取的"异常路径上的链路"作为可疑定位
                sample['drift_location'] = list(label_result['derived_drift_location'])
            
            if drift_params:
                sample['drift_params'] = drift_params
            
            # 元信息
            sample['topology'] = topo_name
            sample['scheduling_config'] = sched_config_id
            
            # SAFLA-style baseline 所需字段:
            # 记录该实验开始时的初始 routing (declared intent path set),
            # 用于后续对比 I vs Î
            if baseline_routing_str:
                sample['baseline_routing_paths'] = baseline_routing_str
            
            # 同步 snapshot 对象的 label
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
        """
        Args:
            event: drift event 描述
            drift_injector: DriftInjector 实例
            links: 候选链路列表
            topo_data: 拓扑数据
            routing: 当前 routing dict
            intents: 当前实验的意图列表（用于 path drift 选取受影响流）
        """
        drift_params = {
            'event_id': int(event.get('event_id', -1)),
            'event_types': list(event.get('types', [])),
            'duration': int(event.get('duration', 0)),
            'settle_time': int(event.get('settle_time', 0)),
            'gap_before': int(event.get('gap_before', 0)),
        }
        drift_locations = []
        extra_energy_detection = False
        
        # 从 intents 提取意图流的 (src, dst)，用于 path drift 真实重路由
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
                # 方案A: 真实重路由
                # 不再传 target_links（旧版用来在链路上加 1000ms 延迟），
                # 改为传 affected_intent_flows，让 drift_injector 主动改流表
                if not intent_flow_keys:
                    print("  [_inject_event_drifts] WARNING: path drift requested "
                          "but no intent flows available, skipping")
                    continue
                
                path_drift = DriftConfig(
                    drift_type=DriftType.PATH,
                    affected_intent_flows=list(intent_flow_keys),
                )
                ok = drift_injector.inject_drift(path_drift)
                
                # 收集"被影响的链路"用于 drift_location 标注
                # 从 path_drift_backup 里读出新旧路径，把变化的边作为 drift_location
                if ok and drift_injector.path_drift_backup:
                    affected_link_changes = []
                    for flow, old_path in drift_injector.path_drift_backup.items():
                        # 找到对应的新路径
                        for d in drift_injector.active_drifts:
                            if d.get('mode') == 'reroute' and d.get('flow') == flow:
                                new_path = d.get('new_path', [])
                                # 提取新路径上的链路
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
            # 去重，同时尽量保留原始顺序
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
        """
        运行单次实验
        
        v2 改动：
        - 为本次实验生成多个意图
        - 每个快照用每个意图分别标注 → 一个快照产生多个样本
        - 漂移标签自动推导
        """
        # 保存临时拓扑配置
        temp_topo_path = os.path.join(self.output_dir, 'temp_topology.yaml')
        with open(temp_topo_path, 'w') as f:
            yaml.dump(topo_data['config'], f)
        
        # 创建网络环境
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
            # 禁用 IPv6
            print("Disabling IPv6 on all hosts...")
            for host in net.hosts:
                host.cmd('sysctl -w net.ipv6.conf.all.disable_ipv6=1')
                host.cmd('sysctl -w net.ipv6.conf.default.disable_ipv6=1')
            
            print("Waiting for Ryu Controller to sync topology...")
            time.sleep(10)

            # ============================================================
            # 应用路由
            # ============================================================
            print(f"Applying routing configuration: {routing.get('type', 'unknown')}...")
            if not self.routing_applier.apply_routing_from_mininet(routing, net):
                print("Warning: Failed to apply routing via API, will use default forwarding")
            time.sleep(2)
            
            # 验证路由
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

            # 预热网络
            #print("Warming up network...")
            #for i in range(3):
            #    loss = net.pingAll()
            #    print(f"  Warm-up round {i+1}: {loss}% loss")
            #    if loss == 0:
            #        break
            #    time.sleep(2)
            
            # 配置QoS
            qos_config = QoSConfigurator(network_env)
            self._apply_scheduling_config(qos_config, sched_config)
            time.sleep(1)
            
            # 初始化流量回放
            replayer = RealTrafficReplayer(network_env)
            if self.traffic_profile:
                replayer.traffic_profile = self.traffic_profile
            
            # 缩放流量矩阵
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
            
            # 初始化能耗感知采集器
            collector = EnergyAwareCollector(
                network_env,
                self.config.get('controller_url', 'http://127.0.0.1:8080'),
                energy_model=self.energy_model
            )
            collector.set_network(net)
            
            # 传递路由配置给 collector
            if routing and 'paths' in routing:
                all_paths = routing['paths']
                sample_keys = random.sample(list(all_paths.keys()), min(10, len(all_paths)))
                sample_paths = {k: all_paths[k] for k in sample_keys}
                collector.set_configured_routes(sample_paths)
                print(f"  Configured {len(sample_paths)}/{len(all_paths)} route paths for collector (sampled)")

            # 初始化漂移注入器
            drift_injector = DriftInjector(net, self.config.get('controller_url'))
            drift_injector.set_network(net)
            # 关键：让 drift_injector 知道当前的 routing_applier 和 routing 表，
            # 这样 path drift 才能调用 apply_partial_routes 真实改流表
            drift_injector.set_routing_context(self.routing_applier, routing)
            
            # ============================================================
            # 核心改动：先启动流量+采集 baseline，再据此自适应生成意图
            # 避免硬编码意图阈值与仿真实际状态严重不匹配的问题
            # ============================================================
            samples = []
            stochastic_cfg = self.config.get('stochastic_drift', {})
            
            # 先估算总时长，启动 replayer 覆盖整个实验（baseline + 主实验）
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
            
            # 采集 baseline 指标
            baseline = self._collect_baseline_metrics(collector, duration=baseline_dur)
            
            # 用 baseline 计算自适应模板，替换原模板
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
            # 生成意图（基于自适应模板）
            # ============================================================
            hosts = net.hosts
            
            # 修复：先做一次短暂预采集（3秒），看哪些流有实际流量
            # 然后从有流量的流中选取意图目标
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
                
                # 找吞吐最大、路径最长的流
                best_flow = None
                best_score = -1
                for pid, pdata in observed_paths.items():
                    if not isinstance(pdata, dict):
                        continue
                    tput = float(pdata.get('e2e_throughput_mbps', 0) or 0)
                    loss = float(pdata.get('e2e_loss_rate', 0) or 0)
                    nodes = pdata.get('path_nodes', [])
                    nhops = len(nodes) - 1 if nodes else 0
                    
                    # 要求：有流量(>0.1Mbps)、低丢包(<0.5)、路径>=4跳
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
                    # 放宽条件再试一次
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
                        # 打印所有流的情况帮助诊断
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
            
            # 从路由表获取路径（如果预采集没获取到）
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
            
            # ============================================================
            # 修复：用正常阶段实际观测到的 path_nodes 校准 waypoints
            # 确保正常状态下路径一定满足 waypoints 约束
            # 只有漂移注入导致路径变化后才会违约
            # ============================================================
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
                                # 检查当前 waypoints 是否在实际路径中
                                old_wp = intent.path.waypoints
                                missing = [wp for wp in old_wp if wp not in actual_nodes]
                                if missing:
                                    # 从实际路径的中间节点重新选取 waypoints
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
            
            # ============================================================
            # 修复：基于正常采集阶段的实测值校准意图阈值
            # 确保正常阶段 95%+ 样本不违约
            # ============================================================
            if normal_samples and intents:
                print("    [threshold calibration] Calibrating thresholds from normal samples...")
                
                for intent in intents:
                    i_match = getattr(intent, 'match', None) or {}
                    if not isinstance(i_match, dict):
                        i_match = {}
                    i_src = i_match.get('src', '')
                    i_dst = i_match.get('dst', '')
                    
                    # 收集该意图匹配流在正常阶段的所有观测值
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
                    
                    # delay: 阈值 >= 实测P99 × 2
                    if intent.performance and intent.performance.delay and obs_delays:
                        old_th = intent.performance.delay.threshold
                        p99 = float(np.percentile(obs_delays, 99))
                        new_th = max(old_th, p99 * 20)
                        if new_th != old_th:
                            intent.performance.delay.threshold = new_th
                            calibrated.append(f"delay: {old_th:.1f}->{new_th:.1f}ms")
                    
                    # loss: 只看有流量时的loss
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
                    
                    # bandwidth: 阈值 <= 实测有效吞吐P1 × 0.3
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
                    
                    # power: 阈值 >= 实测P99 × 1.1
                    if intent.energy and intent.energy.max_power and obs_powers:
                        old_th = intent.energy.max_power.threshold
                        p99 = float(np.percentile(obs_powers, 99))
                        new_th = max(old_th, p99 * 1.3)
                        if new_th != old_th:
                            intent.energy.max_power.threshold = new_th
                            calibrated.append(f"power: {old_th:.0f}->{new_th:.0f}W")
                    
                    # efficiency: 阈值 <= 实测P1 × 0.3
                    if intent.energy and intent.energy.min_efficiency and obs_effs:
                        old_th = intent.energy.min_efficiency.threshold
                        p1 = float(np.percentile(obs_effs, 1))
                        new_th = min(old_th, p1 * 0.05)
                        if new_th != old_th:
                            intent.energy.min_efficiency.threshold = new_th
                            calibrated.append(f"eff: {old_th:.4f}->{new_th:.6f}")
                    
                    # max_hops: 阈值 >= 实测max + 2
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
                
                # 校准后重新标注正常阶段的样本
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
                
                # 替换
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

            # 清理
            try:
                replayer.stop_replay()
            except Exception as e:
                print(f"Warning: Error stopping replayer:{e}")
            
            # 记录路由信息到所有样本
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
            # 清理路由配置
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
        """从 Mininet 网络中提取链路信息"""
        links = []
        for link in net.links:
            intf1, intf2 = link.intf1, link.intf2
            node1, node2 = intf1.node, intf2.node
        
            # 只关心交换机之间的链路
            if hasattr(node1, 'dpid') and hasattr(node2, 'dpid'):
                links.append({
                    'src': node1.name,
                    'dst': node2.name,
                    'src_port': node1.ports[intf1],
                    'dst_port': node2.ports[intf2]
                })
        return links
    
    def _apply_scheduling_config(self, qos_config, sched_config):
        """应用调度配置"""
        net = qos_config.network_env.net
        
        for switch in net.switches:
            node_conf = sched_config['nodes'].get(switch.name, {})
            policy = node_conf.get('policy', 'FIFO')
            queue_size = node_conf.get('queue_size', 16000)
            weights = node_conf.get('weights')
            
            for intf in switch.intfList():
                if intf.name != 'lo' and not intf.name.startswith('lo'):
                    # 清除现有配置
                    switch.cmd(f'tc qdisc del dev {intf.name} root 2>/dev/null')
                    
                    # 设置队列大小
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
        """运行完整的数据集生成"""
        print("=" * 60)
        print("Real Trace Dataset Generation (v2 - Intent-Driven)")
        print("=" * 60)
        print(f"Drift types: normal(0), performance(1), path(2), energy(3)")
        print(f"Intents per experiment: {self.num_intents_per_experiment}")
        print(f"Labels: AUTO-DERIVED from intent constraints")
        print("=" * 60)
        
        # 准备流量配置
        print("\n[1/5] Preparing traffic profile...")
        self.prepare_traffic_profile()
        
        # 准备拓扑
        print("\n[2/5] Preparing topologies...")
        topologies = self.prepare_topologies()
        
        if not topologies:
            print("Error: No topologies available")
            return
        
        # 生成调度配置
        print("\n[3/5] Generating scheduling configurations...")
        sched_configs = self.generate_scheduling_configs(
            self.config.get('num_scheduling_configs', 10)
        )
        
        # 运行实验
        print("\n[4/5] Running experiments...")
        all_samples = []
        output_path1 = os.path.join(self.output_dir, 'samples_raw_new.jsonl')
        # 关键：清空旧的 samples_raw_new.jsonl，避免与之前实验的样本混杂
        open(output_path1, 'w').close()
        print(f"    Cleared old samples file: {output_path1}")
        available_topos = list(topologies.keys())
        
        for topo_name in available_topos:
            topo_data = topologies[topo_name]
            print(f"\n  Topology: {topo_name}")
            print(f"    Nodes: {topo_data.get('num_nodes', 'N/A')}, "
                  f"Links: {topo_data.get('num_links', 'N/A')}")
            
            # 生成路由变体
            routing_gen = RoutingGenerator(
                self._save_temp_topology(topo_data['config'])
            )
            routings = routing_gen.generate_routing_variants(
                num_variants=min(self.NUM_ROUTING_VARIANTS, self.config.get('num_routings', 5))
            )
            
            # 选择部分调度配置
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
        
        # 保存数据集
        print("\n[5/5] Saving dataset...")
        self._save_dataset(all_samples)
        
        print("\n" + "=" * 60)
        print("Dataset Generation Complete!")
        print("=" * 60)
        print(f"Total samples: {len(all_samples)}")
        print(f"Output directory: {self.output_dir}")
        
        # 打印标签分布
        label_counts = {}
        for s in all_samples:
            dt = s.get('drift_type', 'unknown')
            label_counts[dt] = label_counts.get(dt, 0) + 1
        
        print("\nLabel distribution (auto-derived from intents):")
        for dt, count in sorted(label_counts.items()):
            pct = count / len(all_samples) * 100 if all_samples else 0
            print(f"  {dt}: {count} ({pct:.1f}%)")
        
        # 打印意图多样性统计
        intent_ids = set()
        for s in all_samples:
            intent = s.get('intent', {})
            if intent:
                intent_ids.add(intent.get('intent_id', ''))
        print(f"\nUnique intents used: {len(intent_ids)}")
        
        # 打印"同一状态不同标签"的统计
        self._print_label_diversity_stats(all_samples)
        
        return all_samples
    
    def _print_label_diversity_stats(self, samples):
        """
        统计"同一网络状态在不同意图下得到不同标签"的情况
        这是意图驱动漂移检测与普通异常检测的核心区别
        """
        # 按 timestamp 分组
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
        """
        将独立快照构建为 clause-level 时序预测样本
        
        核心思想：
          输入 = 过去 window_size 个快照的序列
          标签 = 未来 horizon 个快照内每个 clause 是否违约
                 （至少连续 min_persist 个时间步违约才算）
        
        分组 key：(topology, routing_id, sched_config, intent_id)
        每组内按 timestamp 排序后滑窗。
        
        关键过滤：max_gap_seconds
          如果窗口或 future 内相邻快照的时间间隔超过这个阈值，
          说明中间有 sleep / 子实验切换 / 采集断点，
          这种窗口物理上不连续，会让模型学到"实验脚本结构"而非真实趋势，
          所以直接丢弃。
        
        Args:
            samples: 独立快照列表
            window_size: 历史窗口大小
            horizon: 预测未来多少步
            min_persist: 至少连续多少步违约才算漂移
            max_gap_seconds: 相邻快照最大允许时间间隔
        
        Returns:
            sequence_samples: 列表，每个元素包含：
                'window': [snapshot_0, ..., snapshot_{T-1}]
                'intent': intent dict
                'future_clause_labels': {              [clause-level 多标签]
                    'perf': 0/1, 'path': 0/1, 'energy': 0/1
                }
                'future_has_any_drift': bool
                'future_label': int                    向后兼容（单一标签）
                'drift_location': list
                'experiment_id': str
        """
        from collections import defaultdict
        
        # 按 (实验组 + 意图) 分组
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
        n_filtered_gap = 0  # 因时间断点被过滤的窗口数
        
        for group_key, group_samples in groups.items():
            group_samples.sort(key=lambda x: x.get('timestamp', 0))
            
            n = len(group_samples)
            if n < window_size + horizon:
                continue
            
            for i in range(n - window_size - horizon + 1):
                window = group_samples[i : i + window_size]
                future = group_samples[i + window_size : i + window_size + horizon]
                
                # ============================================================
                # 时间断点检查：如果窗口+future 内有相邻快照间隔超过阈值，
                # 说明物理上不连续，丢弃该窗口
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
                # 对每个 clause 独立检查是否有连续 min_persist 个时间步违约
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
                
                # 向后兼容：单一 future_label（优先级 path > perf > energy）
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
                
                # 当前时刻（窗口最后一步）
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
                    # SAFLA-style baseline 需要:从 window 内 snapshot 继承 baseline_routing_paths
                    'baseline_routing_paths': window[0].get('baseline_routing_paths', {}),
                }
                
                # 定位信息：取 future 中第一个 clause 违约的快照位置
                # 如果 future 里没有，回退到 window 末尾找（因为 drift 可能发生在 window 末尾，
                # 而 future 快照可能碰巧落在 normal 段上）
                if future_has_any_drift:
                    for f in future:
                        cl = f.get('clause_labels', {})
                        if any(cl.values()) and f.get('drift_location'):
                            seq_sample['drift_location'] = f['drift_location']
                            break
                    
                    # 回退：future 没找到就去 window 里找（从最近的快照往前）
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
        """
        保存数据集 - 时序预测版本（先窗口再切分）

        关键设计：
        1. 先在所有 raw samples 上构建滑窗序列
        2. 按 (group, window_end_ts) 切分 train/val/test，
           在切分点处留 (window_size + horizon) 的 gap 防止边界泄漏
        
        旧设计（先按 70/15/15 切 raw 再分别滑窗）的问题：
        - val/test 各只有 ~12 个 raw 快照，根本切不出 window=10+horizon=3 的窗口
        - 导致 val/test 永远是空集
        """
        window_size = self.config.get('window_size', 10)
        horizon = self.config.get('prediction_horizon', 3)
        min_persist = self.config.get('min_persist', 2)
        max_gap = self.config.get('max_gap_seconds', 12.0)

        print(f"\nBuilding sequence samples...")
        print(f"  Window size: {window_size}")
        print(f"  Prediction horizon: {horizon}")
        print(f"  Min persistence: {min_persist}")
        print(f"  Max gap between snapshots: {max_gap}s")

        # 1. 先在所有 raw samples 上构建滑窗
        all_seq_samples = self._build_sequences(
            samples, window_size, horizon, min_persist, max_gap_seconds=max_gap
        )
        print(f"  Generated {len(all_seq_samples)} sequence samples from {len(samples)} raw snapshots")

        if not all_seq_samples:
            print("Warning: No sequence samples generated. Saving raw snapshots instead.")
            self._save_dataset_fallback(samples)
            return

        # 2. 按 experiment_id 分组，组内按 window_end_ts 排序，时序切分
        # 在切分点处留 boundary_gap 防止边界窗口泄漏
        from collections import defaultdict
        seq_groups = defaultdict(list)
        for s in all_seq_samples:
            seq_groups[s['experiment_id']].append(s)
        
        boundary_gap = window_size + horizon
        # 每组最少需要多少序列才切分（太少就全给 train）
        min_seqs_for_split = self.config.get('min_seqs_for_split', 10)
        train_samples, val_samples, test_samples = [], [], []
        n_skipped_groups = 0
        
        for exp_id, exp_seqs in seq_groups.items():
            exp_seqs.sort(key=lambda x: x['window_end_ts'])
            n = len(exp_seqs)
            
            if n < min_seqs_for_split:
                # 组太小，全给 train（后面会整体补偿）
                train_samples.extend(exp_seqs)
                n_skipped_groups += 1
                continue
            
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            
            # 自适应 boundary_gap：
            # - 组足够大（≥ 3*boundary_gap）时留严格的 gap 防止边界泄漏
            # - 组较小时留 0 gap，允许轻微边界重叠（总比 val/test 空集强）
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
        
        # 如果 val 或 test 还是空的（所有组都被跳过），做一次全局兜底切分
        if len(val_samples) == 0 or len(test_samples) == 0:
            print(f"  Warning: val or test is empty after per-group split. "
                  f"Performing global fallback split.")
            # 按 window_end_ts 全局排序
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
                'has_baseline_routing_paths': True,  # SAFLA baseline 支持标记
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
        """回退方案：不再随机划分，而是按时间顺序保存原始快照。"""
        train_raw, val_raw, test_raw = self._split_raw_samples_temporally(samples)

        for name, data, fname in [('Train', train_raw, 'train_new.json'),
                                   ('Val', val_raw, 'val_new.json'),
                                   ('Test', test_raw, 'test_new.json')]:
            path = os.path.join(self.output_dir, fname)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"Saved {len(data)} {name} raw snapshot samples (temporal fallback)")

    def _save_temp_topology(self, topo_config):
        """保存临时拓扑配置"""
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
    
    # 默认配置
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
        
        # 能耗相关配置
        'intent_max_power': 1500.0,
        'energy': {
            'switch_chassis_power': 100.0,
            'switch_idle_port_power': 2.0,
            'switch_active_port_power': 5.0,
            'switch_dynamic_power': 10.0,
            'link_transceiver_power': 1.0,
            'link_dynamic_power': 0.5
        },
        
        # 意图生成配置（新增）
        'num_intents_per_experiment': args.num_intents,
        'intent_seed': 42,
        
        'drift_distribution': {
            'normal': 0.4,
            'performance': 0.2,
            'path': 0.2,
            'energy': 0.2
        }
    }
    
    # 加载配置文件
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = yaml.safe_load(f)
            if file_config:
                config.update(file_config)
    
    # 命令行参数覆盖
    if args.output:
        config['output_dir'] = args.output
    
    # 快速测试模式
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
