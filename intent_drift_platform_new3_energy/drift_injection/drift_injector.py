# drift_injection/drift_injector.py


import time
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class DriftType(Enum):

    NORMAL = 0
    PERFORMANCE = 1    # Performance drift
    PATH = 2           # Path drift
    ENERGY = 3         # Energy drift
    MIXED = 4          # Mixed drift

@dataclass
class DriftConfig:
    
    drift_type: DriftType
    
    # Performance drift parameters
    delay_ms: float = 0.0          # Additional delay to inject
    delay_jitter_ms: float = 0.0   # Delay jitter
    loss_rate: float = 0.0         # Packet loss rate (0-1)
    bandwidth_limit_mbps: float = 0.0  # Bandwidth limit
    # Path drift parameters (legacy interface, kept for compatibility)
    affected_flows: List[str] = None  # Affected flows
    new_path: List[str] = None        # New path
    
    # Path drift parameters
    # affected_intent_flows: Affected intent flows, format [(src_host, dst_host), ...]
    # For example [('h1', 'h22')]
    # drift_injector will calculate a different path for these flows and proactively push out the flow tables
    affected_intent_flows: List[Tuple[str, str]] = None
    
    # Energy consumption drift parameters
    energy_mode: str = "suboptimal_routing"  # Energy drift mode
    scatter_traffic: bool = False     # Whether to scatter traffic
    activate_redundant_links: int = 0  # Number of activated redundant links
    device_degradation: Dict[str, float] = None  # Device degradation coefficients
    
    # Target devices
    target_links: List[str] = None
    target_switches: List[str] = None
    
    # Duration
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

    def __init__(self, network=None, controller_url: str = "http://127.0.0.1:8080"):

        self.network = network
        self.controller_url = controller_url
        
        # Record drift caused by injection for recovery purposes
        # Add a ‘node_name’ field to each record so that you know which node to clear when performing a clear operation.
        self.active_drifts: List[dict] = []
        
        # Backup of the raw flow table
        self.original_flows: Dict[str, List[dict]] = {}
        
        # ============================================================
        # Path drift The context required for true re-routing
        # Injected by set_routing_context()
        # ============================================================
        self.routing_applier = None      # RoutingApplier 
        self.current_routing = None      # Current routing dict
        # Record the “replaced path stream” each time path drift is injected; this is used to restore the path when clearing.
        self.path_drift_backup: Dict[Tuple[str, str], List[str]] = {}
        
    def set_network(self, network):

        self.network = network
    
    def set_routing_context(self, routing_applier, current_routing: Dict):

        self.routing_applier = routing_applier
        self.current_routing = current_routing
        if current_routing and 'paths' in current_routing:
            print(f"  [DriftInjector] Routing context loaded: "
                  f"{len(current_routing['paths'])} flows available for path drift")

    # ------------------------------------------------------------------
    # Core Utility: Execute tc commands within the node namespace
    # ------------------------------------------------------------------

    def _run_tc_on_node(self, node_name: str, tc_cmd: str) -> bool:

        if not self.network:
            print("  [tc] No network available")
            return False

        try:
            node = self.network.get(node_name)
        except Exception:
            print(f"  [tc] Cannot find node '{node_name}' in network")
            return False

        # node.cmd() executes a command synchronously within the node's netns and returns the output
        output = node.cmd(tc_cmd)

        # When the `tc` command fails, it outputs to `stderr`, and `node.cmd()` incorporates this into the return value.
        if output and any(kw in output.lower() for kw in ('error', 'cannot', 'invalid', 'unknown')):
            print(f"  [tc] Warning on {node_name}: {output.strip()}")
            # Does not return False: Some warnings (such as “quantum of class is big”) are harmless
        return True

    def _clear_tc_on_node(self, node_name: str, interface: str):
        """Delete tc rules on the specified node"""
        cmd = f"tc qdisc del dev {interface} root 2>/dev/null"
        self._run_tc_on_node(node_name, cmd)

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def inject_drift(self, config: DriftConfig) -> bool:

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
    # Various types of drift injection
    # ------------------------------------------------------------------

    def _inject_performance_drift(self, config: DriftConfig) -> bool:

        if not self.network:
            print("No network available")
            return False
        
        injected = []
        
        for link_id in config.target_links:
            parts = link_id.split('-')
            if len(parts) != 2:
                continue
            
            src, dst = parts
            
            # Find the interface name and the node it belongs to
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
 
        if self.routing_applier is None or self.current_routing is None:
            print("  [path drift] No routing context set. "
                  "Call set_routing_context() before injecting path drift.")
            return False
        
        all_paths = self.current_routing.get('paths', {})
        if not all_paths:
            print("  [path drift] current_routing has no 'paths'")
            return False
        
        # 1. The stream that decides to change its path
        target_flows = list(config.affected_intent_flows or [])
        
        # If `affected_intent_flows` is not specified, choose any path from `paths`.
        if not target_flows:
            if all_paths:
                target_flows = [list(all_paths.keys())[0]]
                print(f"  [path drift] No affected_intent_flows specified, "
                      f"defaulting to {target_flows[0]}")
        
        # 2. Compute a new path for each target stream
        new_routes = {}  # {(src,dst): new_path}
        
        for flow in target_flows:
            if flow not in all_paths:
                # Perhaps it's a reverse key
                rev = (flow[1], flow[0])
                if rev in all_paths:
                    flow = rev
                else:
                    print(f"  [path drift] flow {flow} not in current routing")
                    continue
            
            current_path = all_paths[flow]
            if not current_path or len(current_path) < 2:
                continue
            
            # Calculate the backup path
            new_path = self._compute_alternative_path(flow[0], flow[1], current_path)
            if not new_path:
                print(f"  [path drift] Cannot find alternative path for {flow}, "
                      f"current={current_path}")
                continue
            
            print(f"  [path drift] {flow}: {current_path} -> {new_path}")
            
            # Record the original path for recovery purposes
            self.path_drift_backup[flow] = list(current_path)
            new_routes[flow] = new_path
        
        if not new_routes:
            print("  [path drift] No routes to change")
            return False
        
        # 3. Call the RoutingApplier to re-distribute the flow table
        ok = self.routing_applier.apply_partial_routes(
            new_routes,
            self.network,
            base_routing=self.current_routing,
        )
        
        if ok:
            for flow, new_path in new_routes.items():
                self.active_drifts.append({
                    'type': 'path',
                    'mode': 'reroute',  # The label is a true reroute
                    'flow': flow,
                    'old_path': self.path_drift_backup.get(flow),
                    'new_path': new_path,
                })
            print(f"  [path drift] Successfully rerouted {len(new_routes)} flow(s)")
            # Waiting for the flow table to take effect
            time.sleep(1.0)
            return True
        else:
            print("  [path drift] apply_partial_routes failed")
            return False
    
    def _compute_alternative_path(self, src_host: str, dst_host: str, 
                                  current_path: List[str]) -> Optional[List[str]]:
        if not self.network:
            return None
        
        # 1. node -> set of neighbor switches
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
        
        # 2. Find the switch to which src_host and dst_host are connected
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
        
        # 3. Mark the “intermediate link” in the current path as disabled to force BFS to take a detour
        forbidden_edges = set()
        for i in range(len(current_path) - 1):
            a, b = current_path[i], current_path[i + 1]
            forbidden_edges.add((a, b))
            forbidden_edges.add((b, a))
        
        # 4. BFS to find the shortest path, avoiding forbidden_edges
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
        
        # 5. Fallback: If “forbidden” is completely disabled and cannot be found, gradually relax the restrictions.
        if not new_path and len(current_path) >= 4:
            mid = len(current_path) // 2
            partial_forbidden = set()
            for i in range(mid - 1, mid + 1):
                if 0 <= i < len(current_path) - 1:
                    a, b = current_path[i], current_path[i + 1]
                    partial_forbidden.add((a, b))
                    partial_forbidden.add((b, a))
            new_path = bfs_avoid(partial_forbidden)
        
        # 6. Verify that the new path is different from the original path
        if new_path and new_path != list(current_path):
            return new_path
        
        return None
    
    def _inject_energy_drift(self, config: DriftConfig) -> bool:

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
        """Injected Mixed Drift"""
        success = True
        
        if config.delay_ms > 0 or config.loss_rate > 0:
            success &= self._inject_performance_drift(config)
        
        if config.energy_mode:
            success &= self._inject_energy_drift(config)
        
        return success
    
    def clear_all_drifts(self) -> bool:
        """Clear all injected drifts"""
        success = True
        
        
        for drift in self.active_drifts:
            try:
                # Skip path drift of the “reroute” type (to be restored uniformly in Step 2)
                if drift.get('mode') == 'reroute':
                    continue
                if drift['type'] in ['performance', 'path', 'energy']:
                    if 'interface' in drift and 'node_name' in drift:
                        self._clear_tc_on_node(drift['node_name'], drift['interface'])
            except Exception as e:
                print(f"  Failed to clear tc drift: {e}")
                success = False
        
        # 2. Handling path drift of the “reroute” type — call `routing_applier` to restore
        reroute_drifts = [d for d in self.active_drifts if d.get('mode') == 'reroute']
        if reroute_drifts and self.routing_applier and self.current_routing:
            try:
                # Collect all streams that need to be restored
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
                    # Waiting for the flow table to take effect
                    time.sleep(0.5)
                else:
                    print(f"  [path drift restore] FAILED")
                    success = False
            except Exception as e:
                print(f"  Failed to restore reroute drifts: {e}")
                success = False
        
        # 3. Clear Status
        self.active_drifts = []
        self.original_flows = {}
        self.path_drift_backup = {}
        
        print("Cleared all drifts")
        return success

    # ------------------------------------------------------------------
    # Internal Tools and Methods
    # ------------------------------------------------------------------
    
    def _build_tc_command(self, interface: str, 
                         delay_ms: float = 0,
                         jitter_ms: float = 0,
                         loss_rate: float = 0,
                         bandwidth_mbps: float = 0) -> str:
        """Build the tc command string (without node information, caller decides where to execute)"""
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
        """Backward-compatible interface lookup (returns only the interface name)"""
        intf, _ = self._find_interface_and_node(src, dst)
        return intf
    
    def _get_all_links(self) -> List[str]:
        """Get all inter-switch link IDs (excluding host links)"""
        if not self.network:
            return []
        
        links = []
        for link in self.network.links:
            src = link.intf1.node.name
            dst = link.intf2.node.name
            # Return only inter-switch links (names beginning with ‘s’)
            if src.startswith('s') and dst.startswith('s'):
                links.append(f"{src}-{dst}")
        
        return links
    
    def _get_flow_table(self, switch: str) -> List[dict]:
        """Get the flow table of a switch (TODO: implement via REST API)"""
        return []


# ------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------

def create_performance_drift(links: List[str], 
                            delay_ms: float = 50,
                            loss_rate: float = 0.01) -> DriftConfig:
    """Create a performance drift configuration"""
    return DriftConfig(
        drift_type=DriftType.PERFORMANCE,
        target_links=links,
        delay_ms=delay_ms,
        loss_rate=loss_rate
    )


def create_path_drift(switches: List[str], 
                     links_to_disable: List[str]) -> DriftConfig:
    """Create a path drift configuration"""
    return DriftConfig(
        drift_type=DriftType.PATH,
        target_switches=switches,
        target_links=links_to_disable
    )


def create_energy_drift(mode: str = "suboptimal_routing",
                       target_links: List[str] = None,
                       device_degradation: Dict[str, float] = None) -> DriftConfig:

    return DriftConfig(
        drift_type=DriftType.ENERGY,
        energy_mode=mode,
        target_links=target_links or [],
        device_degradation=device_degradation or {}
    )


def create_hidden_energy_drift(links: List[str]) -> DriftConfig:

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
