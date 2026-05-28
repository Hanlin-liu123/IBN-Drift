# ryu_controller/intent_controller.py

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, arp
from ryu.lib import hub
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from webob import Response
import json
import networkx as nx
from collections import defaultdict
import time

simple_switch_instance_name = 'intent_aware_controller'


class IntentAwareController(app_manager.RyuApp):

    
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Basic Network Status
        self.mac_to_port = {}           # {dpid: {mac: port}}
        self.datapaths = {}             # {dpid: datapath}
        self.topology = nx.DiGraph()
        
        self.switch_name_to_dpid = {}   # {'s1': 1, 's2': 2}
        self.dpid_to_switch_name = {}   # {1: 's1', 2: 's2'}
        
        # Host-to-Host Mapping
        self.host_to_switch = {}        # {'h1': 's1'}
        self.host_to_port = {}          # {'h1': 1} - Port number of the host connection
        self.host_to_mac = {}           # {'h1': '00:00:00:00:00:01'}
        self.host_ip_to_name = {}       # {'10.0.0.1': 'h1'}
        self.host_name_to_ip = {}       # {'h1': '10.0.0.1'}
        self.host_mac_to_name = {}      # {'00:00:00:00:00:01': 'h1'}
        
        # Link Mapping
        self.links = {}                 # {(src_dpid, dst_dpid): (src_port, dst_port)}
        
        # Routing Configuration
        self.configured_routes = {}     # {(src_host, dst_host): [dpid_path]}
        self.installed_flows = []       # Record installed flow tables 
        
        # Broadcast Storm Protection
        self._broadcast_history = {}
        
        # Traffic Statistics
        self.port_stats = defaultdict(dict)
        self.flow_stats = defaultdict(dict)
        
        # Intent Management
        self.intents = {}
        
        # Monitoring Thread
        self.monitor_thread = hub.spawn(self._monitor_loop)
        
        # Register REST API
        wsgi = kwargs['wsgi']
        wsgi.register(IntentRestController, {simple_switch_instance_name: self})
        
        self.logger.info("=" * 50)
        self.logger.info("IntentAwareController initialized (Full Fix)")
        self.logger.info("REST API: http://0.0.0.0:8080")
        self.logger.info("=" * 50)
    
    # ==================== Router Configuration API ====================
    
    def configure_routing_variant(self, config):

        try:
            # 1. Clear old configurations
            self._clear_all_configured_flows()
            self.configured_routes.clear()
            self.installed_flows.clear()
            
            # 2. Configure Topology Mapping
            if 'topology_mapping' in config:
                self._configure_topology_mapping(config['topology_mapping'])
            
            # 3. Configure the link
            if 'links' in config:
                self._configure_links(config['links'])
            
            # 4. Configure and install the router
            if 'routes' in config:
                for route_config in config['routes']:
                    self._configure_and_install_route(
                        route_config['src'],
                        route_config['dst'],
                        route_config['path']
                    )
            
            self.logger.info(f"Routing variant configured: {len(self.configured_routes)} routes")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to configure routing: {e}")
            return False
    
    def _configure_topology_mapping(self, mapping):
        """Configure an accurate topological map"""
        # Switch Mapping
        if 'switches' in mapping:
            self.switch_name_to_dpid = mapping['switches'].copy()
            self.dpid_to_switch_name = {v: k for k, v in mapping['switches'].items()}
        
        # Host Mapping
        if 'hosts' in mapping:
            for host_name, host_info in mapping['hosts'].items():
                if isinstance(host_info, dict):
                   
                    self.host_to_switch[host_name] = host_info.get('switch')
                    self.host_to_port[host_name] = host_info.get('port')
                    self.host_to_mac[host_name] = host_info.get('mac')
                    
                    ip = host_info.get('ip')
                    if ip:
                        self.host_name_to_ip[host_name] = ip
                        self.host_ip_to_name[ip] = host_name
                    
                    mac = host_info.get('mac')
                    if mac:
                        self.host_mac_to_name[mac] = host_name
                else:
                    
                    self.host_to_switch[host_name] = host_info
        
        self.logger.info(f"Topology mapping: {len(self.switch_name_to_dpid)} switches, "
                        f"{len(self.host_to_switch)} hosts")
    
    def _configure_links(self, links):
        """Configure link information """
        for link in links:
            src_sw = link['src']
            dst_sw = link['dst']
            src_port = link['src_port']
            dst_port = link['dst_port']
            
            src_dpid = self.switch_name_to_dpid.get(src_sw)
            dst_dpid = self.switch_name_to_dpid.get(dst_sw)
            
            if src_dpid and dst_dpid:
                # Two-way link
                self.links[(src_dpid, dst_dpid)] = (src_port, dst_port)
                self.links[(dst_dpid, src_dpid)] = (dst_port, src_port)
                
                # Add to topology diagram
                self.topology.add_edge(src_dpid, dst_dpid, port=src_port)
                self.topology.add_edge(dst_dpid, src_dpid, port=dst_port)
        
        self.logger.info(f"Links configured: {len(links)} links")
    
    def _configure_and_install_route(self, src_host, dst_host, path):
        """Configure and install a single route """
        # Convert path to DPIDs
        dpid_path = []
        for sw in path:
            dpid = self.switch_name_to_dpid.get(sw)
            if dpid is None:
                try:
                    dpid = int(sw.replace('s', ''))
                except:
                    self.logger.error(f"Unknown switch: {sw}")
                    return False
            dpid_path.append(dpid)
        
        # Save the routing configuration
        self.configured_routes[(src_host, dst_host)] = dpid_path
        
        # Retrieve information about the target host
        dst_ip = self.host_name_to_ip.get(dst_host)
        dst_switch = self.host_to_switch.get(dst_host)
        dst_port = self.host_to_port.get(dst_host)  # Exact Port
        
        if not dst_ip:
            self.logger.warning(f"No IP for host {dst_host}, cannot install flow")
            return False
        
        # Install flow tables on every switch along the path
        for i, curr_dpid in enumerate(dpid_path):
            if curr_dpid not in self.datapaths:
                continue
            
            datapath = self.datapaths[curr_dpid]
            parser = datapath.ofproto_parser
            
            # Select the output port
            if i < len(dpid_path) - 1:
                # Intermediate node: Forward to the next hop
                next_dpid = dpid_path[i + 1]
                link_key = (curr_dpid, next_dpid)
                if link_key not in self.links:
                    self.logger.error(f"No link from {curr_dpid} to {next_dpid}")
                    continue
                out_port = self.links[link_key][0]
            else:
                # The Final Hop: Forwarding to the Host
                dst_dpid = self.switch_name_to_dpid.get(dst_switch)
                if curr_dpid != dst_dpid:
                    self.logger.error(f"Path ends at {curr_dpid} but host is on {dst_dpid}")
                    continue
                
                if dst_port is None:
                    self.logger.error(f"No port info for host {dst_host}")
                    continue
                out_port = dst_port
            
            # Install a flow table based on the IPv4 destination address
            match = parser.OFPMatch(
                eth_type=0x0800,  # IPv4
                ipv4_dst=dst_ip
            )
            actions = [parser.OFPActionOutput(out_port)]
            
            # Use high priority to ensure that MAC learning rules are overridden
            self._add_flow(datapath, priority=100, match=match, actions=actions)
            
            # Record installed flow meters
            self.installed_flows.append({
                'dpid': curr_dpid,
                'switch': self.dpid_to_switch_name.get(curr_dpid),
                'match': {'ipv4_dst': dst_ip},
                'out_port': out_port,
                'route': f"{src_host}->{dst_host}",
                'path_index': i
            })
        
        self.logger.info(f"Route installed: {src_host} -> {dst_host} via {path}")
        return True
    
    def _clear_all_configured_flows(self):
        """Clear all configured flow tables"""
        for flow_info in self.installed_flows:
            dpid = flow_info['dpid']
            if dpid in self.datapaths:
                datapath = self.datapaths[dpid]
                parser = datapath.ofproto_parser
                ofproto = datapath.ofproto
                
                # Delete the high-priority flow table
                match = parser.OFPMatch(
                    eth_type=0x0800,
                    ipv4_dst=flow_info['match']['ipv4_dst']
                )
                mod = parser.OFPFlowMod(
                    datapath=datapath,
                    command=ofproto.OFPFC_DELETE,
                    out_port=ofproto.OFPP_ANY,
                    out_group=ofproto.OFPG_ANY,
                    priority=100,
                    match=match
                )
                datapath.send_msg(mod)
        
        self.installed_flows.clear()
    
    def clear_routes(self):
        """Clear all route configurations"""
        self._clear_all_configured_flows()
        self.configured_routes.clear()
        self.logger.info("All routes cleared")
        return True
    
    # ==================== Verification API====================
    
    def verify_routes(self):
        """Verify whether the routing configuration is active"""
        verification = {
            'configured_routes': {},
            'installed_flows': [],
            'issues': []
        }
        
        # Check the configured routes
        for (src, dst), path in self.configured_routes.items():
            path_names = [self.dpid_to_switch_name.get(d, str(d)) for d in path]
            verification['configured_routes'][f"{src}->{dst}"] = path_names
        
        # Check the installed flow meters
        for flow in self.installed_flows:
            verification['installed_flows'].append({
                'switch': flow['switch'],
                'match': flow['match'],
                'out_port': flow['out_port'],
                'route': flow['route']
            })
        
        # Check for potential issues
        for (src, dst), path in self.configured_routes.items():
            # Check whether every switch along the path is connected
            for dpid in path:
                if dpid not in self.datapaths:
                    verification['issues'].append(
                        f"Switch {dpid} in route {src}->{dst} not connected"
                    )
            
            # Check for the corresponding IP mapping
            if dst not in self.host_name_to_ip:
                verification['issues'].append(
                    f"No IP mapping for destination {dst}"
                )
            
            # Check the port for the last hop
            if dst not in self.host_to_port:
                verification['issues'].append(
                    f"No port mapping for host {dst}"
                )
        
        verification['is_valid'] = len(verification['issues']) == 0
        return verification
    
    # ==================== Switch Event Handling ====================
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Switch connection handling"""
        datapath = ev.msg.datapath
        dpid = datapath.id
        
        self.datapaths[dpid] = datapath
        self.topology.add_node(dpid)
        
        # Automatically create switch name mappings
        switch_name = f's{dpid}'
        if switch_name not in self.switch_name_to_dpid:
            self.switch_name_to_dpid[switch_name] = dpid
            self.dpid_to_switch_name[dpid] = switch_name
        
        # Install the default flow table (lowest priority)
        self._install_default_flow(datapath)
        
        self.logger.info(f"Switch {dpid} ({switch_name}) connected")
    
    def _install_default_flow(self, datapath):
        """Install the default flow table"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(
            ofproto.OFPP_CONTROLLER,
            ofproto.OFPCML_NO_BUFFER
        )]
        self._add_flow(datapath, 0, match, actions)
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Packet Processing (Prioritize route configuration; use MAC learning as a fallback)"""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        dpid = datapath.id
        
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        
        if eth is None:
            return
        
        # Ignore LLDP and IPv6 multicast
        if eth.ethertype == 0x88cc:
            return
        if eth.dst.startswith('33:33'):
            return
        
        dst = eth.dst
        src = eth.src
        
        # Broadcast Storm Protection
        if dst == 'ff:ff:ff:ff:ff:ff':
            broadcast_key = (src, dpid)
            current_time = time.time()
            self._broadcast_history = {
                k: v for k, v in self._broadcast_history.items()
                if current_time - v < 1.0
            }
            if broadcast_key in self._broadcast_history:
                return
            self._broadcast_history[broadcast_key] = current_time
        
        # Learning MAC
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port
        
        # Select the output port
        # If a route is configured and the packet is an IPv4 packet, the route should have already been processed by the flow table.
        # This section covers: ARP, traffic with unconfigured routes, and the first packet before route configuration.
        
        if dst in self.mac_to_port.get(dpid, {}):
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD
        
        actions = [parser.OFPActionOutput(out_port)]
        
        # Install low-priority MAC rules only for traffic not routed by configuration
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            # Using a lower priority will not override high-priority rules configured in the routing table
            self._add_flow(datapath, 1, match, actions, idle_timeout=300)
        
        # Send the packet
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=in_port,
            actions=actions,
            data=msg.data
        )
        datapath.send_msg(out)
    
    # ==================== Flow Table Management ====================
    
    def _add_flow(self, datapath, priority, match, actions, idle_timeout=0, hard_timeout=0):
        """Add a flow entry"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout
        )
        datapath.send_msg(mod)
    
    # ==================== Monitoring ====================
    
    def _monitor_loop(self):
        """Monitoring loop"""
        while True:
            for dp in list(self.datapaths.values()):
                self._request_stats(dp)
            hub.sleep(2)
    
    def _request_stats(self, datapath):
        """Request Statistics"""
        parser = datapath.ofproto_parser
        
        req = parser.OFPPortStatsRequest(datapath, 0, datapath.ofproto.OFPP_ANY)
        datapath.send_msg(req)
        
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)
    
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        """Handle port statistics reply"""
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        
        for stat in body:
            self.port_stats[dpid][stat.port_no] = {
                'rx_packets': stat.rx_packets,
                'tx_packets': stat.tx_packets,
                'rx_bytes': stat.rx_bytes,
                'tx_bytes': stat.tx_bytes,
                'rx_dropped': stat.rx_dropped,
                'tx_dropped': stat.tx_dropped,
                'rx_errors': stat.rx_errors,
                'tx_errors': stat.tx_errors
            }
    
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        """Handle flow statistics reply"""
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        
        self.flow_stats[dpid] = []
        for stat in body:
            self.flow_stats[dpid].append({
                'priority': stat.priority,
                'packet_count': stat.packet_count,
                'byte_count': stat.byte_count,
                'duration_sec': stat.duration_sec
            })
    
    # ==================== Network State Query ====================
    
    def get_network_state(self):
        """Get the current network state"""
        return {
            'switches': list(self.datapaths.keys()),
            'switch_names': self.dpid_to_switch_name,
            'host_mappings': {
                'host_to_switch': self.host_to_switch,
                'host_to_port': self.host_to_port,
                'host_to_ip': self.host_name_to_ip
            },
            'links': {f"{self.dpid_to_switch_name.get(k[0], k[0])}->{self.dpid_to_switch_name.get(k[1], k[1])}": 
                     {'src_port': v[0], 'dst_port': v[1]} 
                     for k, v in self.links.items()},
            'configured_routes': {
                f"{k[0]}->{k[1]}": [self.dpid_to_switch_name.get(d, d) for d in v]
                for k, v in self.configured_routes.items()
            },
            'installed_flows_count': len(self.installed_flows),
            'port_stats': {str(k): dict(v) for k, v in self.port_stats.items()},
            'flow_stats': {str(k): v for k, v in self.flow_stats.items()},
            'mac_table': {str(k): dict(v) for k, v in self.mac_to_port.items()}
        }
    
    # ==================== Intent Management ====================
    
    def install_intent(self, intent_config):
        """Install an intent"""
        intent_id = intent_config.get('intent_id', f'intent_{len(self.intents)}')
        self.intents[intent_id] = intent_config
        self.logger.info(f"Intent installed: {intent_id}")
        return True
    
    def remove_intent(self, intent_id):
        """Remove an intent"""
        if intent_id in self.intents:
            del self.intents[intent_id]
            return True
        return False
    
    def check_intent(self, intent_id):
        """Verify compliance with intent"""
        if intent_id not in self.intents:
            return {'compliant': False, 'reason': 'Intent not found'}
        return {'compliant': True, 'violations': []}


class IntentRestController(ControllerBase):
    
    
    def __init__(self, req, link, data, **config):
        super().__init__(req, link, data, **config)
        self.controller = data[simple_switch_instance_name]
    
    def _make_response(self, data, status=200):
        body = json.dumps(data, default=str).encode('utf-8')
        response = Response()
        response.content_type = 'application/json'
        response.charset = 'utf-8'
        response.status = status
        response.body = body
        return response
    
    @route('state', '/state', methods=['GET'])
    def get_state(self, req, **kwargs):
        """GET /state - Get network state"""
        return self._make_response(self.controller.get_network_state())
    
    @route('routing', '/routing', methods=['POST'])
    def configure_routing(self, req, **kwargs):
        """POST /routing - Configure routing variant (complete configuration)"""
        try:
            body = json.loads(req.body.decode('utf-8'))
            success = self.controller.configure_routing_variant(body)
            return self._make_response({'success': success})
        except Exception as e:
            self.logger.error(f"Error in configure_routing: {e}")
            return self._make_response({'success': False, 'error': str(e)}, 500)
    
    @route('routes_clear', '/routes', methods=['DELETE'])
    def clear_routes(self, req, **kwargs):
        """DELETE /routes - Clear all routes"""
        success = self.controller.clear_routes()
        return self._make_response({'success': success})
    
    @route('verify', '/verify', methods=['GET'])
    def verify_routes(self, req, **kwargs):
        """GET /verify - Verify route configuration"""
        result = self.controller.verify_routes()
        return self._make_response(result)
    
    @route('intent_post', '/intent', methods=['POST'])
    def install_intent(self, req, **kwargs):
        """POST /intent - Install an intent"""
        try:
            body = json.loads(req.body.decode('utf-8'))
            success = self.controller.install_intent(body)
            return self._make_response({'success': success})
        except Exception as e:
            return self._make_response({'success': False, 'error': str(e)}, 500)
    
    @route('intents_list', '/intents', methods=['GET'])
    def list_intents(self, req, **kwargs):
        """GET /intents - List all intents"""
        return self._make_response(self.controller.intents)
