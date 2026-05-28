# utils/routing_applier.py
"""
路由配置应用器 - 完整修复版

修复内容：
- 问题A/B：从 Mininet 精确获取 host-switch-port-mac-ip 映射
- 问题C：完整传递 topology_mapping + links + routes
- 问题E：添加验证功能，确认路由真正生效
"""
import requests
import json
import time
from typing import Dict, List, Optional, Any


class RoutingApplier:
    """将路由配置应用到 Controller - 完整修复版"""
    
    def __init__(self, controller_url: str = 'http://127.0.0.1:8080'):
        self.controller_url = controller_url.rstrip('/')
        self.current_routing = None
        self.last_verification = None
    
    def apply_routing_from_mininet(self, routing_config: Dict, net) -> bool:
        """
        从 Mininet 网络对象精确获取拓扑信息并应用路由
        
        Args:
            routing_config: 路由配置 {'type': '...', 'paths': {(src,dst): [path]}}
            net: Mininet 网络对象
        
        Returns:
            bool: 是否成功
        """
        # 1. 从 Mininet 精确提取拓扑信息（修复问题A/B）
        topology_mapping = self._extract_topology_from_mininet(net)
        
        # 2. 从 Mininet 精确提取链路信息（修复问题C）
        links = self._extract_links_from_mininet(net)
        
        # 3. 转换路由格式
        routes = self._convert_routes(routing_config)
        
        # 4. 构建完整配置
        full_config = {
            'topology_mapping': topology_mapping,
            'links': links,
            'routes': routes
        }
        
        # 5. 发送到 Controller
        return self._send_routing_config(full_config, routing_config)
    
    def _extract_topology_from_mininet(self, net) -> Dict:
        """从 Mininet 精确提取拓扑映射（修复问题A/B）"""
        mapping = {
            'switches': {},
            'hosts': {}
        }
        
        # 提取交换机信息
        for switch in net.switches:
            name = switch.name
            dpid = switch.dpid
            if dpid:
                # dpid 可能是十六进制字符串
                if isinstance(dpid, str):
                    dpid = int(dpid, 16)
                mapping['switches'][name] = dpid
            else:
                # 从名称推断 dpid
                try:
                    mapping['switches'][name] = int(name.replace('s', ''))
                except:
                    pass
        
        # 提取主机信息（精确映射，修复问题A/B）
        for host in net.hosts:
            host_name = host.name
            host_info = {
                'ip': host.IP(),
                'mac': host.MAC()
            }
            
            # 找到主机连接的交换机和端口
            for intf in host.intfList():
                if intf.link:
                    # 找到链路另一端（交换机）
                    link = intf.link
                    if link.intf1.node == host:
                        peer_intf = link.intf2
                    else:
                        peer_intf = link.intf1
                    
                    peer_node = peer_intf.node
                    if hasattr(peer_node, 'dpid'):  # 是交换机
                        host_info['switch'] = peer_node.name
                        # 获取交换机侧的端口号
                        host_info['port'] = peer_node.ports.get(peer_intf)
                        break
            
            mapping['hosts'][host_name] = host_info
        
        return mapping
    
    def _extract_links_from_mininet(self, net) -> List[Dict]:
        """从 Mininet 精确提取链路信息（修复问题C）"""
        links = []
        seen = set()
        
        for link in net.links:
            intf1, intf2 = link.intf1, link.intf2
            node1, node2 = intf1.node, intf2.node
            
            # 只处理交换机之间的链路
            if not (hasattr(node1, 'dpid') and hasattr(node2, 'dpid')):
                continue
            
            # 避免重复（双向链路只记录一次）
            link_key = tuple(sorted([node1.name, node2.name]))
            if link_key in seen:
                continue
            seen.add(link_key)
            
            # 获取端口号
            port1 = node1.ports.get(intf1)
            port2 = node2.ports.get(intf2)
            
            if port1 is not None and port2 is not None:
                links.append({
                    'src': node1.name,
                    'dst': node2.name,
                    'src_port': port1,
                    'dst_port': port2
                })
        
        return links
    
    def _convert_routes(self, routing_config: Dict) -> List[Dict]:
        """转换路由格式"""
        routes = []
        paths = routing_config.get('paths', {})
        
        for (src, dst), path in paths.items():
            if path:
                routes.append({
                    'src': src,
                    'dst': dst,
                    'path': path
                })
        
        return routes
    
    def _send_routing_config(self, full_config: Dict, original_config: Dict) -> bool:
        """发送路由配置到 Controller"""
        try:
            response = requests.post(
                f'{self.controller_url}/routing',
                json=full_config,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    self.current_routing = original_config
                    print(f"Applied routing: {original_config.get('type', 'custom')} "
                          f"({len(full_config.get('routes', []))} routes)")
                    return True
            
            print(f"Failed to apply routing: {response.text}")
            return False
            
        except requests.exceptions.ConnectionError:
            print(f"Cannot connect to controller at {self.controller_url}")
            return False
        except Exception as e:
            print(f"Error applying routing: {e}")
            return False
    
    def apply_routing(self, routing_config: Dict, topology_info: Dict = None) -> bool:
        """
        应用路由配置（兼容旧接口，但使用完整配置）
        
        注意：推荐使用 apply_routing_from_mininet() 以获得精确映射
        """
        try:
            # 清除旧路由
            self.clear_routes()
            time.sleep(0.5)
            
            # 构建完整配置
            full_config = {
                'routes': self._convert_routes(routing_config)
            }
            
            # 如果提供了拓扑信息，进行转换
            if topology_info:
                full_config['topology_mapping'] = self._convert_topology_info(topology_info)
                full_config['links'] = topology_info.get('links', [])
            
            return self._send_routing_config(full_config, routing_config)
            
        except Exception as e:
            print(f"Error applying routing: {e}")
            return False
    
    def _convert_topology_info(self, topology_info: Dict) -> Dict:
        """转换拓扑信息格式（兼容旧格式）"""
        mapping = {
            'switches': {},
            'hosts': {}
        }
        
        # 交换机
        for sw in topology_info.get('switches', []):
            if isinstance(sw, str):
                try:
                    dpid = int(sw.replace('s', ''))
                    mapping['switches'][sw] = dpid
                except:
                    pass
        
        # 主机（尝试从完整信息构建）
        hosts = topology_info.get('hosts', [])
        host_details = topology_info.get('host_details', {})
        
        for host in hosts:
            if host in host_details:
                mapping['hosts'][host] = host_details[host]
            else:
                # 推断（不推荐，但保持兼容）
                try:
                    host_num = int(host.replace('h', ''))
                    mapping['hosts'][host] = {
                        'switch': f's{host_num}',
                        'port': 1,
                        'ip': f'10.0.0.{host_num}',
                        'mac': f'00:00:00:00:00:{host_num:02x}'
                    }
                except:
                    pass
        
        return mapping
    
    def clear_routes(self) -> bool:
        """清除所有路由配置"""
        try:
            response = requests.delete(
                f'{self.controller_url}/routes',
                timeout=5
            )
            self.current_routing = None
            return response.status_code == 200
        except:
            return False
    
    # ============================================================
    # 方案A: 真实重路由所需的部分路由更新接口
    # ============================================================
    
    def apply_partial_routes(self, partial_routes: Dict, net,
                             base_routing: Dict) -> bool:
        """
        部分流路由更新（只换指定流的路径，其他流保持不变）
        
        实现：因为 Ryu controller 只支持全量配置下发（没有增量 API），
        这里把 base_routing 的所有流复制一份，把 partial_routes 中
        指定的流替换成新路径，再整体下发。
        
        Args:
            partial_routes: 要更新的路由 {(src_host, dst_host): [s1, s5, s22], ...}
            net: Mininet 网络对象
            base_routing: 当前的完整 routing dict
        
        Returns:
            bool: 是否成功
        """
        if not base_routing or 'paths' not in base_routing:
            print("[apply_partial_routes] base_routing missing 'paths'")
            return False
        
        # 1. 构造合并后的 paths（从 base_routing 复制，再用 partial_routes 覆盖）
        merged_paths = dict(base_routing['paths'])  # 浅拷贝
        n_changed = 0
        for flow, new_path in partial_routes.items():
            if flow in merged_paths:
                merged_paths[flow] = list(new_path)
                n_changed += 1
            else:
                # 反向 key
                rev = (flow[1], flow[0])
                if rev in merged_paths:
                    merged_paths[rev] = list(reversed(new_path))
                    n_changed += 1
                else:
                    # 新流：直接加进去
                    merged_paths[flow] = list(new_path)
                    n_changed += 1
        
        print(f"[apply_partial_routes] Merging {n_changed} flow override(s) "
              f"into base routing ({len(base_routing['paths'])} flows total)")
        
        # 2. 构造合并后的完整 routing config
        merged_routing = dict(base_routing)
        merged_routing['paths'] = merged_paths
        
        # 3. 提取拓扑、链路、转换 routes（重用现有代码）
        topology_mapping = self._extract_topology_from_mininet(net)
        links = self._extract_links_from_mininet(net)
        routes = self._convert_routes(merged_routing)
        
        full_config = {
            'topology_mapping': topology_mapping,
            'links': links,
            'routes': routes,
        }
        
        # 4. 全量下发
        return self._send_routing_config(full_config, merged_routing)
    
    def restore_routes(self, flows_to_restore: List, net,
                       base_routing: Dict) -> bool:
        """
        把指定流恢复到 base_routing 中的原始路径
        
        实现：直接全量重新下发 base_routing 即可——因为 base_routing 就是
        原始路径，所以下发 base_routing 等价于"恢复所有被改的流"。
        
        Args:
            flows_to_restore: 要恢复的流列表 [(src_host, dst_host), ...]
                              （目前仅用于日志，因为下面会全量恢复）
            net: Mininet 网络对象
            base_routing: 原始的完整 routing dict
        
        Returns:
            bool: 是否成功
        """
        if not base_routing or 'paths' not in base_routing:
            print("[restore_routes] base_routing missing 'paths'")
            return False
        
        print(f"[restore_routes] Restoring {len(flows_to_restore)} flow(s) "
              f"by re-applying base routing")
        
        # 直接全量重新下发原始 routing
        topology_mapping = self._extract_topology_from_mininet(net)
        links = self._extract_links_from_mininet(net)
        routes = self._convert_routes(base_routing)
        
        full_config = {
            'topology_mapping': topology_mapping,
            'links': links,
            'routes': routes,
        }
        
        return self._send_routing_config(full_config, base_routing)
    
    def verify_routing(self) -> Dict:
        """验证路由配置是否生效（修复问题E）"""
        try:
            response = requests.get(
                f'{self.controller_url}/verify',
                timeout=5
            )
            if response.status_code == 200:
                self.last_verification = response.json()
                return self.last_verification
        except Exception as e:
            print(f"Verification failed: {e}")
        
        return {'is_valid': False, 'error': 'Failed to verify'}
    
    def get_network_state(self) -> Optional[Dict]:
        """获取当前网络状态"""
        try:
            response = requests.get(
                f'{self.controller_url}/state',
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def wait_for_controller(self, timeout: int = 30) -> bool:
        """等待 Controller 就绪"""
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = requests.get(
                    f'{self.controller_url}/state',
                    timeout=2
                )
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
        return False


def apply_routing_to_controller(
    routing_config: Dict,
    net=None,
    controller_url: str = 'http://127.0.0.1:8080',
    topology_info: Dict = None
) -> bool:
    """便捷函数：应用路由到 Controller"""
    applier = RoutingApplier(controller_url)
    
    if net is not None:
        # 推荐：从 Mininet 精确获取信息
        return applier.apply_routing_from_mininet(routing_config, net)
    else:
        # 兼容：使用提供的拓扑信息
        return applier.apply_routing(routing_config, topology_info)
