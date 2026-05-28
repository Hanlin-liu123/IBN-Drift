# utils/routing_applier.py

import requests
import json
import time
from typing import Dict, List, Optional, Any


class RoutingApplier:
    """Apply the routing configuration to the Controller """
    
    def __init__(self, controller_url: str = 'http://127.0.0.1:8080'):
        self.controller_url = controller_url.rstrip('/')
        self.current_routing = None
        self.last_verification = None
    
    def apply_routing_from_mininet(self, routing_config: Dict, net) -> bool:

        # 1. Extracting Topological Information Precisely from Mininet
        topology_mapping = self._extract_topology_from_mininet(net)
        
        # 2. Extracting link information precisely from Mininet
        links = self._extract_links_from_mininet(net)
        
        # 3. Convert route format
        routes = self._convert_routes(routing_config)
        
        # 4. Build a complete configuration
        full_config = {
            'topology_mapping': topology_mapping,
            'links': links,
            'routes': routes
        }
        
        # 5. Send to Controller
        return self._send_routing_config(full_config, routing_config)
    
    def _extract_topology_from_mininet(self, net) -> Dict:
        """Accurately extracting topology maps from Mininet"""
        mapping = {
            'switches': {},
            'hosts': {}
        }
        
        # Retrieve switch information
        for switch in net.switches:
            name = switch.name
            dpid = switch.dpid
            if dpid:
                # dpid may be a hexadecimal string
                if isinstance(dpid, str):
                    dpid = int(dpid, 16)
                mapping['switches'][name] = dpid
            else:
                # Deducing the dpid from the name
                try:
                    mapping['switches'][name] = int(name.replace('s', ''))
                except:
                    pass
        
        # Retrieve host information
        for host in net.hosts:
            host_name = host.name
            host_info = {
                'ip': host.IP(),
                'mac': host.MAC()
            }
            
            # Find the switch and port to which the host is connected
            for intf in host.intfList():
                if intf.link:
                    # Locate the other end of the link (the switch)
                    link = intf.link
                    if link.intf1.node == host:
                        peer_intf = link.intf2
                    else:
                        peer_intf = link.intf1
                    
                    peer_node = peer_intf.node
                    if hasattr(peer_node, 'dpid'):  # It's a switch
                        host_info['switch'] = peer_node.name
                        # Get the port number on the switch side
                        host_info['port'] = peer_node.ports.get(peer_intf)
                        break
            
            mapping['hosts'][host_name] = host_info
        
        return mapping
    
    def _extract_links_from_mininet(self, net) -> List[Dict]:
        """Extracting link information precisely from Mininet"""
        links = []
        seen = set()
        
        for link in net.links:
            intf1, intf2 = link.intf1, link.intf2
            node1, node2 = intf1.node, intf2.node
            
            # Only process links between switches
            if not (hasattr(node1, 'dpid') and hasattr(node2, 'dpid')):
                continue
            
            # Avoid duplicates (record bidirectional links only once)
            link_key = tuple(sorted([node1.name, node2.name]))
            if link_key in seen:
                continue
            seen.add(link_key)
            
            # Get the port numbers
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
        """Converting route format"""
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
        """Sending routing configuration to Controller"""
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

        try:
            # Remove the old router
            self.clear_routes()
            time.sleep(0.5)
            
            # Build a complete configuration
            full_config = {
                'routes': self._convert_routes(routing_config)
            }
            
            # If topology information is provided, perform the conversion
            if topology_info:
                full_config['topology_mapping'] = self._convert_topology_info(topology_info)
                full_config['links'] = topology_info.get('links', [])
            
            return self._send_routing_config(full_config, routing_config)
            
        except Exception as e:
            print(f"Error applying routing: {e}")
            return False
    
    def _convert_topology_info(self, topology_info: Dict) -> Dict:
        """Converting topology information format (compatible with old format)"""
        mapping = {
            'switches': {},
            'hosts': {}
        }
        
        # Switch
        for sw in topology_info.get('switches', []):
            if isinstance(sw, str):
                try:
                    dpid = int(sw.replace('s', ''))
                    mapping['switches'][sw] = dpid
                except:
                    pass
        
        # Host (attempting to reconstruct from the available information)
        hosts = topology_info.get('hosts', [])
        host_details = topology_info.get('host_details', {})
        
        for host in hosts:
            if host in host_details:
                mapping['hosts'][host] = host_details[host]
            else:
                # Inferred (not recommended, but maintained for compatibility)
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
        """Clear all routing configurations"""
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
    # Option A: The routing update interfaces required for true re-routing
    # ============================================================
    
    def apply_partial_routes(self, partial_routes: Dict, net,
                             base_routing: Dict) -> bool:

        if not base_routing or 'paths' not in base_routing:
            print("[apply_partial_routes] base_routing missing 'paths'")
            return False
        
        # 1. Construct the merged paths (copied from `base_routing` and overridden by `partial_routes`)
        merged_paths = dict(base_routing['paths'])  # Shallow copy
        n_changed = 0
        for flow, new_path in partial_routes.items():
            if flow in merged_paths:
                merged_paths[flow] = list(new_path)
                n_changed += 1
            else:
                # Reverse key
                rev = (flow[1], flow[0])
                if rev in merged_paths:
                    merged_paths[rev] = list(reversed(new_path))
                    n_changed += 1
                else:
                    # New stream: Add directly
                    merged_paths[flow] = list(new_path)
                    n_changed += 1
        
        print(f"[apply_partial_routes] Merging {n_changed} flow override(s) "
              f"into base routing ({len(base_routing['paths'])} flows total)")
        
        # 2. Construct the merged full routing config
        merged_routing = dict(base_routing)
        merged_routing['paths'] = merged_paths
        
        # 3. Extract topology, links, and convert routes (reusing existing code)
        topology_mapping = self._extract_topology_from_mininet(net)
        links = self._extract_links_from_mininet(net)
        routes = self._convert_routes(merged_routing)
        
        full_config = {
            'topology_mapping': topology_mapping,
            'links': links,
            'routes': routes,
        }
        
        # 4. Full-scale rollout
        return self._send_routing_config(full_config, merged_routing)
    
    def restore_routes(self, flows_to_restore: List, net,
                       base_routing: Dict) -> bool:

        if not base_routing or 'paths' not in base_routing:
            print("[restore_routes] base_routing missing 'paths'")
            return False
        
        print(f"[restore_routes] Restoring {len(flows_to_restore)} flow(s) "
              f"by re-applying base routing")
        
        # Directly re-distribute the entire original routing
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
        """Verify if the routing configuration is effective"""
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
        """Get the current network state"""
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
        """Wait for the Controller to be ready"""
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
    """Utility Functions: Application Routing to Controller"""
    applier = RoutingApplier(controller_url)
    
    if net is not None:
        # Recommended: Extract information precisely from Mininet
        return applier.apply_routing_from_mininet(routing_config, net)
    else:
        # Compatible: Use the provided topology information
        return applier.apply_routing(routing_config, topology_info)
