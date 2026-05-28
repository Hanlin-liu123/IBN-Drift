# utils/routing_generator.py
"""
路由配置生成器 - 修复版

修复内容：
1. 返回格式包含 id, type, paths
2. paths 的 key 是 (src_host, dst_host) 元组
3. paths 的 value 是交换机路径列表
4. 支持主机到主机的路径计算
"""
import networkx as nx
import random
import yaml
import os
from typing import Dict, List, Tuple, Optional, Any


class RoutingGenerator:
    """路由配置生成器 - 生成多种路由变体"""
    
    def __init__(self, topology_config):
        """
        初始化路由生成器
        
        Args:
            topology_config: 拓扑配置文件路径或配置字典
        """
        if isinstance(topology_config, str):
            self.topology = self._load_topology(topology_config)
        else:
            self.topology = topology_config
            
        self.graph = self._build_graph()
        self.switch_graph = self._build_switch_graph()
        
        # 主机到交换机的映射
        self.host_to_switch: Dict[str, str] = {}
        self.switch_to_hosts: Dict[str, List[str]] = {}
        self._build_host_switch_mapping()
    
    def _load_topology(self, config_path: str) -> dict:
        """加载拓扑配置"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _build_graph(self) -> nx.Graph:
        """构建完整拓扑图（包含主机和交换机）"""
        G = nx.Graph()
        
        # 添加节点
        for node in self.topology.get('nodes', []):
            node_id = node.get('id') or node.get('name')
            G.add_node(node_id, **node)
        
        # 添加交换机节点（如果单独定义）
        for switch in self.topology.get('switches', []):
            switch_id = switch.get('id') or switch.get('name')
            G.add_node(switch_id, type='switch', **switch)
        
        # 添加主机节点（如果单独定义）
        for host in self.topology.get('hosts', []):
            host_id = host.get('id') or host.get('name')
            G.add_node(host_id, type='host', **host)
        
        # 添加链路
        for link in self.topology.get('links', []):
            src = link.get('src') or link.get('source')
            dst = link.get('dst') or link.get('target')
            bandwidth = link.get('bandwidth', link.get('capacity', 100))
            delay = link.get('delay', 1)
            
            G.add_edge(src, dst, bandwidth=bandwidth, delay=delay)
        
        return G
    
    def _build_switch_graph(self) -> nx.Graph:
        """构建仅包含交换机的拓扑图"""
        G = nx.Graph()
        
        # 识别交换机节点
        switches = set()
        
        # 从 nodes 中识别交换机
        for node in self.topology.get('nodes', []):
            node_id = node.get('id') or node.get('name')
            if node_id.startswith('s') or node.get('type') == 'switch':
                switches.add(node_id)
                G.add_node(node_id)
        
        # 从 switches 中添加
        for switch in self.topology.get('switches', []):
            switch_id = switch.get('id') or switch.get('name')
            switches.add(switch_id)
            G.add_node(switch_id)
        
        # 添加交换机之间的链路
        for link in self.topology.get('links', []):
            src = link.get('src') or link.get('source')
            dst = link.get('dst') or link.get('target')
            
            if src in switches and dst in switches:
                bandwidth = link.get('bandwidth', link.get('capacity', 100))
                delay = link.get('delay', 1)
                G.add_edge(src, dst, bandwidth=bandwidth, delay=delay)
        
        return G
    
    def _build_host_switch_mapping(self):
        """构建主机-交换机映射"""
        # 从 hosts 配置中获取
        for host in self.topology.get('hosts', []):
            host_id = host.get('id') or host.get('name')
            switch_id = host.get('switch') or host.get('connected_to')
            
            if switch_id:
                self.host_to_switch[host_id] = switch_id
                if switch_id not in self.switch_to_hosts:
                    self.switch_to_hosts[switch_id] = []
                self.switch_to_hosts[switch_id].append(host_id)
        
        # 如果没有明确的 hosts 配置，从链路中推断
        if not self.host_to_switch:
            for link in self.topology.get('links', []):
                src = link.get('src') or link.get('source')
                dst = link.get('dst') or link.get('target')
                
                # 假设 h* 是主机，s* 是交换机
                if src.startswith('h') and dst.startswith('s'):
                    self.host_to_switch[src] = dst
                    if dst not in self.switch_to_hosts:
                        self.switch_to_hosts[dst] = []
                    self.switch_to_hosts[dst].append(src)
                elif dst.startswith('h') and src.startswith('s'):
                    self.host_to_switch[dst] = src
                    if src not in self.switch_to_hosts:
                        self.switch_to_hosts[src] = []
                    self.switch_to_hosts[src].append(dst)
        
        # 如果还是没有，假设每个交换机连接一个同编号的主机
        if not self.host_to_switch:
            for node in self.switch_graph.nodes():
                if node.startswith('s'):
                    host_id = 'h' + node[1:]
                    self.host_to_switch[host_id] = node
                    self.switch_to_hosts[node] = [host_id]
    
    def get_hosts(self) -> List[str]:
        """获取所有主机列表"""
        return list(self.host_to_switch.keys())
    
    def get_switches(self) -> List[str]:
        """获取所有交换机列表"""
        return list(self.switch_graph.nodes())
    
    def _get_switch_path(self, src_switch: str, dst_switch: str, weight: str = 'delay') -> Optional[List[str]]:
        """获取两个交换机之间的路径"""
        if src_switch == dst_switch:
            return [src_switch]
        
        try:
            return nx.shortest_path(self.switch_graph, src_switch, dst_switch, weight=weight)
        except nx.NetworkXNoPath:
            return None
    
    def _get_host_path(self, src_host: str, dst_host: str, weight: str = 'delay') -> Optional[List[str]]:
        """获取两个主机之间的交换机路径"""
        src_switch = self.host_to_switch.get(src_host)
        dst_switch = self.host_to_switch.get(dst_host)
        
        if not src_switch or not dst_switch:
            return None
        
        return self._get_switch_path(src_switch, dst_switch, weight)
    
    def generate_shortest_path_routing(self) -> Dict[Tuple[str, str], List[str]]:
        """
        生成最短路径路由
        
        Returns:
            {(src_host, dst_host): [switch_path], ...}
        """
        paths = {}
        hosts = self.get_hosts()
        
        for src in hosts:
            for dst in hosts:
                if src != dst:
                    path = self._get_host_path(src, dst)
                    if path:
                        paths[(src, dst)] = path
        
        return paths
    
    def generate_perturbed_routing(self, base_paths: Dict[Tuple[str, str], List[str]], 
                                    perturbation_ratio: float = 0.1) -> Dict[Tuple[str, str], List[str]]:
        """
        生成扰动路由
        
        Args:
            base_paths: 基础路由
            perturbation_ratio: 扰动比例
        
        Returns:
            扰动后的路由
        """
        perturbed = {}
        
        for (src, dst), base_path in base_paths.items():
            if random.random() < perturbation_ratio and base_path:
                # 尝试找到替代路径
                src_switch = self.host_to_switch.get(src)
                dst_switch = self.host_to_switch.get(dst)
                
                if src_switch and dst_switch:
                    try:
                        all_paths = list(nx.all_simple_paths(
                            self.switch_graph, src_switch, dst_switch,
                            cutoff=len(base_path) + 2
                        ))
                        if len(all_paths) > 1:
                            # 选择一个不同的路径
                            alt_paths = [p for p in all_paths if p != base_path]
                            if alt_paths:
                                perturbed[(src, dst)] = random.choice(alt_paths)
                            else:
                                perturbed[(src, dst)] = base_path
                        else:
                            perturbed[(src, dst)] = base_path
                    except:
                        perturbed[(src, dst)] = base_path
                else:
                    perturbed[(src, dst)] = base_path
            else:
                perturbed[(src, dst)] = base_path
        
        return perturbed
    
    def generate_ecmp_routing(self) -> Dict[Tuple[str, str], List[str]]:
        """
        生成 ECMP（等价多路径）路由
        对于有多条等价路径的情况，随机选择一条
        """
        paths = {}
        hosts = self.get_hosts()
        
        for src in hosts:
            for dst in hosts:
                if src != dst:
                    src_switch = self.host_to_switch.get(src)
                    dst_switch = self.host_to_switch.get(dst)
                    
                    if src_switch and dst_switch and src_switch != dst_switch:
                        try:
                            # 获取所有最短路径
                            all_shortest = list(nx.all_shortest_paths(
                                self.switch_graph, src_switch, dst_switch
                            ))
                            if all_shortest:
                                paths[(src, dst)] = random.choice(all_shortest)
                        except:
                            path = self._get_host_path(src, dst)
                            if path:
                                paths[(src, dst)] = path
                    elif src_switch == dst_switch:
                        paths[(src, dst)] = [src_switch]
        
        return paths
    
    def generate_waypoint_routing(self, waypoint_switch: str) -> Dict[Tuple[str, str], List[str]]:
        """
        生成经过指定中间节点的路由
        
        Args:
            waypoint_switch: 必经交换机
        """
        paths = {}
        hosts = self.get_hosts()
        
        for src in hosts:
            for dst in hosts:
                if src != dst:
                    src_switch = self.host_to_switch.get(src)
                    dst_switch = self.host_to_switch.get(dst)
                    
                    if src_switch and dst_switch:
                        try:
                            # 路径 = src -> waypoint -> dst
                            path1 = nx.shortest_path(self.switch_graph, src_switch, waypoint_switch)
                            path2 = nx.shortest_path(self.switch_graph, waypoint_switch, dst_switch)
                            
                            # 合并路径（去掉重复的 waypoint）
                            full_path = path1 + path2[1:]
                            paths[(src, dst)] = full_path
                        except:
                            # 如果无法经过 waypoint，使用最短路径
                            path = self._get_host_path(src, dst)
                            if path:
                                paths[(src, dst)] = path
        
        return paths
    
    def generate_routing_variants(self, num_variants: int = 100, 
                                   output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        生成多种路由变体
        
        Args:
            num_variants: 变体数量
            output_dir: 输出目录（可选）
        
        Returns:
            路由变体列表，每个变体格式：
            {
                'id': 0,
                'type': 'shortest_path',
                'paths': {('h1', 'h2'): ['s1', 's3', 's2'], ...}
            }
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 生成基础最短路径路由
        base_paths = self.generate_shortest_path_routing()
        variants = []
        
        for i in range(num_variants):
            if i == 0:
                # 第一个变体：纯最短路径
                paths = base_paths
                routing_type = 'shortest_path'
            elif i < 5:
                # ECMP 路由
                paths = self.generate_ecmp_routing()
                routing_type = 'ecmp'
            elif i < 20:
                # 轻微扰动 (5%)
                paths = self.generate_perturbed_routing(base_paths, 0.05)
                routing_type = 'perturbed_light'
            elif i < 50:
                # 中等扰动 (15%)
                paths = self.generate_perturbed_routing(base_paths, 0.15)
                routing_type = 'perturbed_medium'
            elif i < 80:
                # 较大扰动 (25%)
                paths = self.generate_perturbed_routing(base_paths, 0.25)
                routing_type = 'perturbed_heavy'
            else:
                # Waypoint 路由
                switches = self.get_switches()
                if switches:
                    waypoint = random.choice(switches)
                    paths = self.generate_waypoint_routing(waypoint)
                    routing_type = f'waypoint_{waypoint}'
                else:
                    paths = base_paths
                    routing_type = 'shortest_path'
            
            # 构建标准格式的路由变体
            variant = {
                'id': i,
                'type': routing_type,
                'paths': paths,
                'num_paths': len(paths),
                'description': f'Routing variant {i}: {routing_type}'
            }
            
            # 保存到文件（可选）
            if output_dir:
                routing_file = os.path.join(output_dir, f'routing_{i:03d}.yaml')
                # 转换 tuple keys 为字符串以便 YAML 序列化
                serializable_paths = {f"{k[0]}->{k[1]}": v for k, v in paths.items()}
                save_data = {
                    'id': i,
                    'type': routing_type,
                    'paths': serializable_paths
                }
                with open(routing_file, 'w') as f:
                    yaml.dump(save_data, f)
            
            variants.append(variant)
        
        print(f"Generated {num_variants} routing variants")
        print(f"  - Hosts: {len(self.get_hosts())}")
        print(f"  - Switches: {len(self.get_switches())}")
        print(f"  - Paths per variant: {len(base_paths)}")
        
        return variants
    
    def get_path_for_hosts(self, src_host: str, dst_host: str, 
                           routing_variant: Dict[str, Any]) -> Optional[List[str]]:
        """
        从路由变体中获取指定主机对的路径
        
        Args:
            src_host: 源主机
            dst_host: 目标主机
            routing_variant: 路由变体
        
        Returns:
            交换机路径列表
        """
        paths = routing_variant.get('paths', {})
        return paths.get((src_host, dst_host))


# 便捷函数
def generate_routing_for_topology(topology_config, num_variants: int = 10) -> List[Dict[str, Any]]:
    """便捷函数：为拓扑生成路由变体"""
    generator = RoutingGenerator(topology_config)
    return generator.generate_routing_variants(num_variants)


if __name__ == '__main__':
    # 测试
    import sys
    
    if len(sys.argv) > 1:
        topo_file = sys.argv[1]
    else:
        # 创建测试拓扑
        test_topo = {
            'nodes': [
                {'id': 's1'}, {'id': 's2'}, {'id': 's3'},
                {'id': 's4'}, {'id': 's5'}
            ],
            'hosts': [
                {'id': 'h1', 'switch': 's1'},
                {'id': 'h2', 'switch': 's2'},
                {'id': 'h3', 'switch': 's3'},
                {'id': 'h4', 'switch': 's4'},
                {'id': 'h5', 'switch': 's5'}
            ],
            'links': [
                {'src': 's1', 'dst': 's2', 'bandwidth': 100, 'delay': 1},
                {'src': 's1', 'dst': 's3', 'bandwidth': 100, 'delay': 2},
                {'src': 's2', 'dst': 's4', 'bandwidth': 100, 'delay': 1},
                {'src': 's3', 'dst': 's4', 'bandwidth': 100, 'delay': 1},
                {'src': 's3', 'dst': 's5', 'bandwidth': 100, 'delay': 1},
                {'src': 's4', 'dst': 's5', 'bandwidth': 100, 'delay': 2},
                {'src': 'h1', 'dst': 's1', 'bandwidth': 100, 'delay': 0},
                {'src': 'h2', 'dst': 's2', 'bandwidth': 100, 'delay': 0},
                {'src': 'h3', 'dst': 's3', 'bandwidth': 100, 'delay': 0},
                {'src': 'h4', 'dst': 's4', 'bandwidth': 100, 'delay': 0},
                {'src': 'h5', 'dst': 's5', 'bandwidth': 100, 'delay': 0}
            ]
        }
        
        # 保存测试拓扑
        with open('/tmp/test_topo.yaml', 'w') as f:
            yaml.dump(test_topo, f)
        topo_file = '/tmp/test_topo.yaml'
    
    print(f"Testing RoutingGenerator with {topo_file}")
    generator = RoutingGenerator(topo_file)
    
    print(f"\nHosts: {generator.get_hosts()}")
    print(f"Switches: {generator.get_switches()}")
    print(f"Host-Switch mapping: {generator.host_to_switch}")
    
    # 生成路由变体
    variants = generator.generate_routing_variants(num_variants=5)
    
    print("\n--- Routing Variants ---")
    for v in variants:
        print(f"\nVariant {v['id']}: {v['type']}")
        print(f"  Paths: {v['num_paths']}")
        # 显示几个示例路径
        for (src, dst), path in list(v['paths'].items())[:3]:
            print(f"    {src} -> {dst}: {path}")
