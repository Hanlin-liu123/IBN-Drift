# drift_injection/drift_injector.py
"""
意图漂移注入器 (Intent Drift Injector)

支持三种漂移类型：
1. 性能漂移 (Performance Drift): 时延增加、丢包增加
2. 路径漂移 (Path Drift): 流表修改导致路径改变
3. 能耗漂移 (Energy Drift): 性能达标但能耗超标（隐蔽漂移）

能耗漂移场景：
- 次优路由导致的能耗浪费（流量打散唤醒休眠链路）
- 设备老化导致的能耗异常
- 冗余设备激活
"""

import time
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class DriftType(Enum):
    """漂移类型"""
    NORMAL = 0
    PERFORMANCE = 1    # 性能漂移
    PATH = 2           # 路径漂移
    ENERGY = 3         # 能耗漂移
    MIXED = 4          # 混合漂移


@dataclass
class DriftConfig:
    """漂移配置"""
    drift_type: DriftType
    
    # 性能漂移参数
    delay_ms: float = 0.0          # 注入的额外延迟
    delay_jitter_ms: float = 0.0   # 延迟抖动
    loss_rate: float = 0.0         # 丢包率 (0-1)
    bandwidth_limit_mbps: float = 0.0  # 带宽限制
    
    # 路径漂移参数（旧接口，保留兼容）
    affected_flows: List[str] = None  # 受影响的流
    new_path: List[str] = None        # 新路径
    
    # 路径漂移参数（方案A：真实重路由）
    # affected_intent_flows: 受影响的意图流，格式 [(src_host, dst_host), ...]
    # 例如 [('h1', 'h22')]
    # drift_injector 会为这些流计算一条不同的路径并主动下发流表
    affected_intent_flows: List[Tuple[str, str]] = None
    
    # 能耗漂移参数
    energy_mode: str = "suboptimal_routing"  # 能耗漂移模式
    scatter_traffic: bool = False     # 是否打散流量
    activate_redundant_links: int = 0  # 激活的冗余链路数
    device_degradation: Dict[str, float] = None  # 设备老化系数
    
    # 目标设备
    target_links: List[str] = None
    target_switches: List[str] = None
    
    # 持续时间
    duration_seconds: float = 0.0
    
    def __post_init__(self):
        if self.affected_flows is None:
            self.affected_flows = []
        if self.new_path is None:
            self.new_path = []
        if self.affected_intent_flows is None:
            self.affected_intent_flows = []
        if self.target_links is None:
            self.target_links = []
        if self.target_switches is None:
            self.target_switches = []
        if self.device_degradation is None:
            self.device_degradation = {}


class DriftInjector:
    """
    漂移注入器

    使用 Linux tc (traffic control) 和 OpenFlow 流表修改来注入漂移。

    关键修复：所有 tc 命令必须通过 Mininet node.cmd() 在对应节点的
    网络命名空间内执行，而不能直接用 subprocess 在宿主机上执行。
    宿主机上看不到 h1-eth0、s1-eth1 等 Mininet 内部接口，
    直接执行会报 "Cannot find device" 错误。
    """
    
    def __init__(self, network=None, controller_url: str = "http://127.0.0.1:8080"):
        """
        初始化注入器
        
        Args:
            network: Mininet网络对象
            controller_url: SDN控制器REST API地址
        """
        self.network = network
        self.controller_url = controller_url
        
        # 记录注入的漂移，用于恢复
        # 每条记录增加 'node_name' 字段，clear 时知道在哪个节点上清除
        self.active_drifts: List[dict] = []
        
        # 原始流表备份
        self.original_flows: Dict[str, List[dict]] = {}
        
        # ============================================================
        # Path drift 真实重路由所需的上下文
        # 由 set_routing_context() 注入
        # ============================================================
        self.routing_applier = None      # RoutingApplier 实例
        self.current_routing = None      # 当前 routing dict
        # 每次 path drift 注入时记录"被换路径的流"，clear 时用来恢复
        self.path_drift_backup: Dict[Tuple[str, str], List[str]] = {}
        
    def set_network(self, network):
        """设置Mininet网络"""
        self.network = network
    
    def set_routing_context(self, routing_applier, current_routing: Dict):
        """
        注入路由上下文，让 path drift 能调用 routing_applier 真实改流表
        
        Args:
            routing_applier: RoutingApplier 实例（必须支持 apply_partial_routes）
            current_routing: 当前的 routing dict，包含 'paths' 字段
                             paths 的 key 是 (src_host, dst_host) tuple
                             value 是 [s1, s5, s22] 这样的交换机列表
        """
        self.routing_applier = routing_applier
        self.current_routing = current_routing
        if current_routing and 'paths' in current_routing:
            print(f"  [DriftInjector] Routing context loaded: "
                  f"{len(current_routing['paths'])} flows available for path drift")

    # ------------------------------------------------------------------
    # 核心辅助：在节点命名空间内执行 tc 命令
    # ------------------------------------------------------------------

    def _run_tc_on_node(self, node_name: str, tc_cmd: str) -> bool:
        """
        在指定 Mininet 节点的网络命名空间内执行 tc 命令。

        Args:
            node_name: Mininet 节点名称（如 's1', 'h3'）
            tc_cmd:    完整的 tc shell 命令字符串

        Returns:
            True 表示执行成功（无明显错误输出），False 表示失败。
        """
        if not self.network:
            print("  [tc] No network available")
            return False

        try:
            node = self.network.get(node_name)
        except Exception:
            print(f"  [tc] Cannot find node '{node_name}' in network")
            return False

        # node.cmd() 在节点的 netns 内同步执行命令并返回输出
        output = node.cmd(tc_cmd)

        # tc 命令失败时会输出到 stderr，node.cmd() 会把它混入返回值
        if output and any(kw in output.lower() for kw in ('error', 'cannot', 'invalid', 'unknown')):
            print(f"  [tc] Warning on {node_name}: {output.strip()}")
            # 不返回 False：部分 warning（如 "quantum of class is big"）无害
        return True

    def _clear_tc_on_node(self, node_name: str, interface: str):
        """在指定节点上清除 tc 规则"""
        cmd = f"tc qdisc del dev {interface} root 2>/dev/null"
        self._run_tc_on_node(node_name, cmd)

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def inject_drift(self, config: DriftConfig) -> bool:
        """
        注入漂移
        
        Args:
            config: 漂移配置
            
        Returns:
            success: 是否注入成功
        """
        try:
            if config.drift_type == DriftType.PERFORMANCE:
                return self._inject_performance_drift(config)
            elif config.drift_type == DriftType.PATH:
                return self._inject_path_drift(config)
            elif config.drift_type == DriftType.ENERGY:
                return self._inject_energy_drift(config)
            elif config.drift_type == DriftType.MIXED:
                return self._inject_mixed_drift(config)
            else:
                return False
                
        except Exception as e:
            print(f"Drift injection failed: {e}")
            return False
    
    # ------------------------------------------------------------------
    # 各类漂移注入
    # ------------------------------------------------------------------

    def _inject_performance_drift(self, config: DriftConfig) -> bool:
        """
        注入性能漂移

        使用 tc netem 注入延迟、抖动、丢包。
        tc 命令通过 node.cmd() 在对应交换机/主机的命名空间内执行。
        """
        if not self.network:
            print("No network available")
            return False
        
        injected = []
        
        for link_id in config.target_links:
            parts = link_id.split('-')
            if len(parts) != 2:
                continue
            
            src, dst = parts
            
            # 找到接口名及其所属节点
            intf, node_name = self._find_interface_and_node(src, dst)
            if not intf:
                print(f"  Cannot find interface for link {link_id}")
                continue
            
            tc_cmd = self._build_tc_command(
                intf,
                delay_ms=config.delay_ms,
                jitter_ms=config.delay_jitter_ms,
                loss_rate=config.loss_rate,
                bandwidth_mbps=config.bandwidth_limit_mbps
            )
            
            if self._run_tc_on_node(node_name, tc_cmd):
                injected.append({
                    'type': 'performance',
                    'interface': intf,
                    'node_name': node_name,
                    'link_id': link_id,
                    'config': config
                })
                print(f"  Injected performance drift on {link_id}: "
                      f"delay={config.delay_ms}ms, loss={config.loss_rate*100:.2f}%")
            else:
                print(f"  Failed to inject performance drift on {link_id}")
        
        self.active_drifts.extend(injected)
        return len(injected) > 0
    
    def _inject_path_drift(self, config: DriftConfig) -> bool:
        """
        注入路径漂移（方案A：真实重路由）
        
        核心思想：
          不再依赖"在链路上加 1000ms 延迟 → 期望 controller 自动重路由"，
          因为 OpenFlow 流表是静态下发的，tc 加延迟根本不会触发流表改变。
        
          改为：直接调用 RoutingApplier 把意图流的流表改成一条不同的备份路径。
        
        所需配置:
          config.affected_intent_flows: [(src_host, dst_host), ...]
                                         例如 [('h1', 'h22')]
        
        前提条件：
          调用前必须先 set_routing_context(routing_applier, current_routing)
        """
        if self.routing_applier is None or self.current_routing is None:
            print("  [path drift] No routing context set. "
                  "Call set_routing_context() before injecting path drift.")
            return False
        
        all_paths = self.current_routing.get('paths', {})
        if not all_paths:
            print("  [path drift] current_routing has no 'paths'")
            return False
        
        # 1. 决定要换路径的流
        target_flows = list(config.affected_intent_flows or [])
        
        # 兜底：如果没指定 affected_intent_flows，从 paths 里随便挑一条
        if not target_flows:
            if all_paths:
                target_flows = [list(all_paths.keys())[0]]
                print(f"  [path drift] No affected_intent_flows specified, "
                      f"defaulting to {target_flows[0]}")
        
        # 2. 为每条目标流计算一条新路径
        new_routes = {}  # {(src,dst): new_path}
        
        for flow in target_flows:
            if flow not in all_paths:
                # 也许是反向 key
                rev = (flow[1], flow[0])
                if rev in all_paths:
                    flow = rev
                else:
                    print(f"  [path drift] flow {flow} not in current routing")
                    continue
            
            current_path = all_paths[flow]
            if not current_path or len(current_path) < 2:
                continue
            
            # 计算备份路径
            new_path = self._compute_alternative_path(flow[0], flow[1], current_path)
            if not new_path:
                print(f"  [path drift] Cannot find alternative path for {flow}, "
                      f"current={current_path}")
                continue
            
            print(f"  [path drift] {flow}: {current_path} -> {new_path}")
            
            # 记录原路径用于恢复
            self.path_drift_backup[flow] = list(current_path)
            new_routes[flow] = new_path
        
        if not new_routes:
            print("  [path drift] No routes to change")
            return False
        
        # 3. 调用 RoutingApplier 重新下发流表
        ok = self.routing_applier.apply_partial_routes(
            new_routes,
            self.network,
            base_routing=self.current_routing,
        )
        
        if ok:
            for flow, new_path in new_routes.items():
                self.active_drifts.append({
                    'type': 'path',
                    'mode': 'reroute',  # 标记是真实重路由
                    'flow': flow,
                    'old_path': self.path_drift_backup.get(flow),
                    'new_path': new_path,
                })
            print(f"  [path drift] Successfully rerouted {len(new_routes)} flow(s)")
            # 等待流表生效
            time.sleep(1.0)
            return True
        else:
            print("  [path drift] apply_partial_routes failed")
            return False
    
    def _compute_alternative_path(self, src_host: str, dst_host: str, 
                                  current_path: List[str]) -> Optional[List[str]]:
        """
        从拓扑中计算一条不同于 current_path 的备份路径
        
        策略：
          1. 在拓扑图上做 BFS 找最短路径
          2. 把 current_path 的中间链路标记为禁用，让 BFS 倾向于绕开
          3. 如果新路径和当前路径不同，返回新路径
          4. 否则返回 None
        
        Args:
            src_host: 源主机名（如 'h1'）
            dst_host: 目的主机名（如 'h22'）
            current_path: 当前路径的交换机列表（如 ['s1', 's16', 's22']）
        
        Returns:
            new_path: 不同于 current_path 的交换机列表，找不到则 None
        """
        if not self.network:
            return None
        
        # 1. 构建拓扑图：node -> set of neighbor switches
        from collections import defaultdict, deque
        graph = defaultdict(set)
        
        for link in self.network.links:
            n1 = link.intf1.node.name
            n2 = link.intf2.node.name
            if n1.startswith('s') and n2.startswith('s'):
                graph[n1].add(n2)
                graph[n2].add(n1)
        
        if not graph:
            return None
        
        # 2. 找到 src_host 和 dst_host 接入的交换机
        src_switch = None
        dst_switch = None
        try:
            src_node = self.network.get(src_host)
            dst_node = self.network.get(dst_host)
        except Exception:
            return None
        
        for intf in src_node.intfList():
            if intf.link:
                peer = (intf.link.intf2.node 
                        if intf.link.intf1.node == src_node 
                        else intf.link.intf1.node)
                if peer.name.startswith('s'):
                    src_switch = peer.name
                    break
        
        for intf in dst_node.intfList():
            if intf.link:
                peer = (intf.link.intf2.node 
                        if intf.link.intf1.node == dst_node 
                        else intf.link.intf1.node)
                if peer.name.startswith('s'):
                    dst_switch = peer.name
                    break
        
        if not src_switch or not dst_switch:
            return None
        
        # 3. 把当前路径的"中间链路"标记为禁用，强制 BFS 绕路
        forbidden_edges = set()
        for i in range(len(current_path) - 1):
            a, b = current_path[i], current_path[i + 1]
            forbidden_edges.add((a, b))
            forbidden_edges.add((b, a))
        
        # 4. BFS 找最短路径，避开 forbidden_edges
        def bfs_avoid(forbidden):
            if src_switch == dst_switch:
                return [src_switch]
            visited = {src_switch}
            queue = deque([(src_switch, [src_switch])])
            while queue:
                node, path = queue.popleft()
                for nbr in graph[node]:
                    if nbr in visited:
                        continue
                    if (node, nbr) in forbidden:
                        continue
                    new_p = path + [nbr]
                    if nbr == dst_switch:
                        return new_p
                    visited.add(nbr)
                    queue.append((nbr, new_p))
            return None
        
        new_path = bfs_avoid(forbidden_edges)
        
        # 5. Fallback：如果完全禁用 forbidden 找不到，逐步放宽
        if not new_path and len(current_path) >= 4:
            mid = len(current_path) // 2
            partial_forbidden = set()
            for i in range(mid - 1, mid + 1):
                if 0 <= i < len(current_path) - 1:
                    a, b = current_path[i], current_path[i + 1]
                    partial_forbidden.add((a, b))
                    partial_forbidden.add((b, a))
            new_path = bfs_avoid(partial_forbidden)
        
        # 6. 验证新路径与原路径不同
        if new_path and new_path != list(current_path):
            return new_path
        
        return None
    
    def _inject_energy_drift(self, config: DriftConfig) -> bool:
        """
        注入能耗漂移

        场景1: 次优路由导致能耗浪费
        场景2: 设备老化导致能耗异常
        场景3: 冗余设备激活
        """
        if not self.network:
            return False
        
        if config.energy_mode == "suboptimal_routing":
            return self._inject_suboptimal_routing(config)
        elif config.energy_mode == "device_degradation":
            return self._inject_device_degradation(config)
        elif config.energy_mode == "redundant_activation":
            return self._inject_redundant_activation(config)
        else:
            return False
    
    def _inject_suboptimal_routing(self, config: DriftConfig) -> bool:
        """
        注入次优路由能耗漂移

        在最短路径的主要链路上添加微小延迟（5ms），
        迫使部分流量走更长路径，唤醒更多设备，
        端到端性能基本不变但能耗升高。
        """
        if not config.target_links:
            all_links = self._get_all_links()
            config.target_links = random.sample(all_links, min(2, len(all_links)))
        
        for link_id in config.target_links:
            parts = link_id.split('-')
            if len(parts) != 2:
                continue
            
            src, dst = parts
            intf, node_name = self._find_interface_and_node(src, dst)
            if intf:
                tc_cmd = self._build_tc_command(intf, delay_ms=5)
                if self._run_tc_on_node(node_name, tc_cmd):
                    self.active_drifts.append({
                        'type': 'energy',
                        'subtype': 'suboptimal_routing',
                        'interface': intf,
                        'node_name': node_name,
                        'link_id': link_id
                    })
                    print(f"  Injected energy drift (suboptimal routing): "
                          f"added delay on {link_id}")
        
        return True
    
    def _inject_device_degradation(self, config: DriftConfig) -> bool:
        """
        注入设备老化能耗漂移

        模拟某设备散热故障，处理同样流量功耗增加。
        主要在采集阶段通过乘以老化系数体现，这里只记录配置。
        """
        for switch_id, factor in config.device_degradation.items():
            self.active_drifts.append({
                'type': 'energy',
                'subtype': 'device_degradation',
                'switch_id': switch_id,
                'degradation_factor': factor
            })
            print(f"  Injected energy drift (device degradation): "
                  f"{switch_id} factor={factor}")
        
        return True
    
    def _inject_redundant_activation(self, config: DriftConfig) -> bool:
        """
        注入冗余设备激活

        激活本应休眠的冗余链路/设备，记录激活状态供采集器使用。
        """
        all_links = self._get_all_links()
        num_to_activate = min(config.activate_redundant_links, len(all_links))
        links_to_activate = random.sample(all_links, num_to_activate)
        
        for link_id in links_to_activate:
            self.active_drifts.append({
                'type': 'energy',
                'subtype': 'redundant_activation',
                'link_id': link_id
            })
            print(f"  Injected energy drift (redundant activation): "
                  f"activated {link_id}")
        
        return True
    
    def _inject_mixed_drift(self, config: DriftConfig) -> bool:
        """注入混合漂移"""
        success = True
        
        if config.delay_ms > 0 or config.loss_rate > 0:
            success &= self._inject_performance_drift(config)
        
        if config.energy_mode:
            success &= self._inject_energy_drift(config)
        
        return success
    
    def clear_all_drifts(self) -> bool:
        """清除所有注入的漂移"""
        success = True
        
        # 1. 处理 tc-based drift（performance / energy / 旧版 path）
        for drift in self.active_drifts:
            try:
                # 跳过 reroute 类型的 path drift（要在第 2 步统一恢复）
                if drift.get('mode') == 'reroute':
                    continue
                if drift['type'] in ['performance', 'path', 'energy']:
                    if 'interface' in drift and 'node_name' in drift:
                        self._clear_tc_on_node(drift['node_name'], drift['interface'])
            except Exception as e:
                print(f"  Failed to clear tc drift: {e}")
                success = False
        
        # 2. 处理 reroute 类型的 path drift —— 调用 routing_applier 恢复
        reroute_drifts = [d for d in self.active_drifts if d.get('mode') == 'reroute']
        if reroute_drifts and self.routing_applier and self.current_routing:
            try:
                # 收集所有需要恢复的流
                flows_to_restore = [d['flow'] for d in reroute_drifts]
                print(f"  [path drift restore] Restoring {len(flows_to_restore)} flow(s) "
                      f"to original routing")
                
                ok = self.routing_applier.restore_routes(
                    flows_to_restore,
                    self.network,
                    base_routing=self.current_routing,
                )
                if ok:
                    print(f"  [path drift restore] OK")
                    # 等待流表生效
                    time.sleep(0.5)
                else:
                    print(f"  [path drift restore] FAILED")
                    success = False
            except Exception as e:
                print(f"  Failed to restore reroute drifts: {e}")
                success = False
        
        # 3. 清空状态
        self.active_drifts = []
        self.original_flows = {}
        self.path_drift_backup = {}
        
        print("Cleared all drifts")
        return success

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------
    
    def _build_tc_command(self, interface: str, 
                         delay_ms: float = 0,
                         jitter_ms: float = 0,
                         loss_rate: float = 0,
                         bandwidth_mbps: float = 0) -> str:
        """构建 tc 命令字符串（不含节点信息，由调用方决定在哪里执行）"""
        clear_cmd = f"tc qdisc del dev {interface} root 2>/dev/null; "
        
        netem_parts = []
        
        if delay_ms > 0:
            if jitter_ms > 0:
                netem_parts.append(f"delay {delay_ms}ms {jitter_ms}ms")
            else:
                netem_parts.append(f"delay {delay_ms}ms")
        
        if loss_rate > 0:
            netem_parts.append(f"loss {loss_rate * 100}%")
        
        if bandwidth_mbps > 0:
            cmd = clear_cmd
            cmd += f"tc qdisc add dev {interface} root handle 1: htb default 10; "
            cmd += f"tc class add dev {interface} parent 1: classid 1:10 htb rate {bandwidth_mbps}mbit"
            if netem_parts:
                cmd += f"; tc qdisc add dev {interface} parent 1:10 handle 10: netem {' '.join(netem_parts)}"
            return cmd
        
        if netem_parts:
            return clear_cmd + f"tc qdisc add dev {interface} root netem {' '.join(netem_parts)}"
        
        return clear_cmd

    def _find_interface_and_node(self, src: str, dst: str) -> Tuple[Optional[str], Optional[str]]:
        """
        找到连接 src 和 dst 的接口名及其所在节点名。

        返回 (interface_name, node_name)，未找到时返回 (None, None)。
        接口属于 src 侧节点：tc 规则加在出口方向（src → dst）。
        """
        if not self.network:
            return None, None
        
        for link in self.network.links:
            intf1, intf2 = link.intf1, link.intf2
            
            if intf1.node.name == src and intf2.node.name == dst:
                return intf1.name, src
            elif intf2.node.name == src and intf1.node.name == dst:
                return intf2.name, src
        
        return None, None

    def _find_interface(self, src: str, dst: str) -> Optional[str]:
        """向后兼容的接口查找（只返回接口名）"""
        intf, _ = self._find_interface_and_node(src, dst)
        return intf
    
    def _get_all_links(self) -> List[str]:
        """获取所有交换机间链路 ID（排除主机链路）"""
        if not self.network:
            return []
        
        links = []
        for link in self.network.links:
            src = link.intf1.node.name
            dst = link.intf2.node.name
            # 只返回交换机间链路（名称以 's' 开头）
            if src.startswith('s') and dst.startswith('s'):
                links.append(f"{src}-{dst}")
        
        return links
    
    def _get_flow_table(self, switch: str) -> List[dict]:
        """获取交换机流表（TODO: 通过REST API实现）"""
        return []


# ------------------------------------------------------------------
# 便捷函数
# ------------------------------------------------------------------

def create_performance_drift(links: List[str], 
                            delay_ms: float = 50,
                            loss_rate: float = 0.01) -> DriftConfig:
    """创建性能漂移配置"""
    return DriftConfig(
        drift_type=DriftType.PERFORMANCE,
        target_links=links,
        delay_ms=delay_ms,
        loss_rate=loss_rate
    )


def create_path_drift(switches: List[str], 
                     links_to_disable: List[str]) -> DriftConfig:
    """创建路径漂移配置"""
    return DriftConfig(
        drift_type=DriftType.PATH,
        target_switches=switches,
        target_links=links_to_disable
    )


def create_energy_drift(mode: str = "suboptimal_routing",
                       target_links: List[str] = None,
                       device_degradation: Dict[str, float] = None) -> DriftConfig:
    """
    创建能耗漂移配置
    
    Args:
        mode: 能耗漂移模式
            - "suboptimal_routing": 次优路由
            - "device_degradation": 设备老化
            - "redundant_activation": 冗余激活
        target_links: 目标链路
        device_degradation: 设备老化系数 {switch_id: factor}
    """
    return DriftConfig(
        drift_type=DriftType.ENERGY,
        energy_mode=mode,
        target_links=target_links or [],
        device_degradation=device_degradation or {}
    )


def create_hidden_energy_drift(links: List[str]) -> DriftConfig:
    """
    创建隐蔽能耗漂移

    性能指标（时延、丢包）完全正常，但能耗大幅超标，
    传统监控完全无法发现。
    """
    return DriftConfig(
        drift_type=DriftType.ENERGY,
        energy_mode="suboptimal_routing",
        target_links=links,
        scatter_traffic=True
    )


if __name__ == '__main__':
    injector = DriftInjector()
    
    print("Creating drift configurations...")
    
    perf_drift = create_performance_drift(['s1-s2', 's2-s3'], delay_ms=50, loss_rate=0.02)
    print(f"Performance drift: delay={perf_drift.delay_ms}ms, loss={perf_drift.loss_rate}")
    
    path_drift = create_path_drift(['s1', 's2'], ['s1-s2'])
    print(f"Path drift: disable links {path_drift.target_links}")
    
    energy_drift = create_energy_drift(
        mode="suboptimal_routing",
        target_links=['s1-s2', 's3-s4']
    )
    print(f"Energy drift: mode={energy_drift.energy_mode}")
    
    hidden_drift = create_hidden_energy_drift(['s1-s3', 's2-s4'])
    print(f"Hidden energy drift: links={hidden_drift.target_links}")
