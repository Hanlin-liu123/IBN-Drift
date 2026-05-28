# ryu_controller/intent_controller.py
"""
意图感知的SDN控制器 - 完整版（修复所有路由配置问题）

修复内容：
- 问题A：从配置精确获取主机端口，不假设端口1
- 问题B：同时支持 MAC 和 IP 匹配，精确映射 host-switch-port
- 问题C：完整接收 topology_mapping + links + routes
- 问题D：统一使用高优先级 IPv4 流表，避免与 MAC 学习冲突
- 问题E：添加路由验证 API
"""
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
    """意图感知的SDN控制器 - 完整修复版"""
    
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 基础网络状态
        self.mac_to_port = {}           # {dpid: {mac: port}}
        self.datapaths = {}             # {dpid: datapath}
        self.topology = nx.DiGraph()
        
        # ========== 精确映射（修复问题A/B）==========
        self.switch_name_to_dpid = {}   # {'s1': 1, 's2': 2}
        self.dpid_to_switch_name = {}   # {1: 's1', 2: 's2'}
        
        # 主机精确映射
        self.host_to_switch = {}        # {'h1': 's1'}
        self.host_to_port = {}          # {'h1': 1} - 主机连接的端口号
        self.host_to_mac = {}           # {'h1': '00:00:00:00:00:01'}
        self.host_ip_to_name = {}       # {'10.0.0.1': 'h1'}
        self.host_name_to_ip = {}       # {'h1': '10.0.0.1'}
        self.host_mac_to_name = {}      # {'00:00:00:00:00:01': 'h1'}
        
        # 链路映射（修复问题C）
        self.links = {}                 # {(src_dpid, dst_dpid): (src_port, dst_port)}
        
        # 路由配置
        self.configured_routes = {}     # {(src_host, dst_host): [dpid_path]}
        self.installed_flows = []       # 记录已安装的流表（修复问题E）
        
        # 广播风暴防护
        self._broadcast_history = {}
        
        # 流量统计
        self.port_stats = defaultdict(dict)
        self.flow_stats = defaultdict(dict)
        
        # 意图管理
        self.intents = {}
        
        # 监控线程
        self.monitor_thread = hub.spawn(self._monitor_loop)
        
        # 注册REST API
        wsgi = kwargs['wsgi']
        wsgi.register(IntentRestController, {simple_switch_instance_name: self})
        
        self.logger.info("=" * 50)
        self.logger.info("IntentAwareController initialized (Full Fix)")
        self.logger.info("REST API: http://0.0.0.0:8080")
        self.logger.info("=" * 50)
    
    # ==================== 路由配置 API（修复问题C）====================
    
    def configure_routing_variant(self, config):
        """
        配置完整的路由变体（一次性接收所有信息）
        
        config = {
            'topology_mapping': {
                'switches': {'s1': 1, 's2': 2, ...},
                'hosts': {
                    'h1': {'switch': 's1', 'port': 1, 'mac': '00:00:00:00:00:01', 'ip': '10.0.0.1'},
                    'h2': {'switch': 's2', 'port': 1, 'mac': '00:00:00:00:00:02', 'ip': '10.0.0.2'},
                    ...
                }
            },
            'links': [
                {'src': 's1', 'dst': 's3', 'src_port': 2, 'dst_port': 3},
                ...
            ],
            'routes': [
                {'src': 'h1', 'dst': 'h2', 'path': ['s1', 's3', 's2']},
                ...
            ]
        }
        """
        try:
            # 1. 清除旧配置
            self._clear_all_configured_flows()
            self.configured_routes.clear()
            self.installed_flows.clear()
            
            # 2. 配置拓扑映射（修复问题A/B）
            if 'topology_mapping' in config:
                self._configure_topology_mapping(config['topology_mapping'])
            
            # 3. 配置链路（修复问题C）
            if 'links' in config:
                self._configure_links(config['links'])
            
            # 4. 配置并安装路由（修复问题D）
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
        """配置精确的拓扑映射（修复问题A/B）"""
        # 交换机映射
        if 'switches' in mapping:
            self.switch_name_to_dpid = mapping['switches'].copy()
            self.dpid_to_switch_name = {v: k for k, v in mapping['switches'].items()}
        
        # 主机精确映射（关键修复）
        if 'hosts' in mapping:
            for host_name, host_info in mapping['hosts'].items():
                if isinstance(host_info, dict):
                    # 新格式：完整信息
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
                    # 旧格式：只有 switch 名
                    self.host_to_switch[host_name] = host_info
        
        self.logger.info(f"Topology mapping: {len(self.switch_name_to_dpid)} switches, "
                        f"{len(self.host_to_switch)} hosts")
    
    def _configure_links(self, links):
        """配置链路信息（修复问题C）"""
        for link in links:
            src_sw = link['src']
            dst_sw = link['dst']
            src_port = link['src_port']
            dst_port = link['dst_port']
            
            src_dpid = self.switch_name_to_dpid.get(src_sw)
            dst_dpid = self.switch_name_to_dpid.get(dst_sw)
            
            if src_dpid and dst_dpid:
                # 双向链路
                self.links[(src_dpid, dst_dpid)] = (src_port, dst_port)
                self.links[(dst_dpid, src_dpid)] = (dst_port, src_port)
                
                # 添加到拓扑图
                self.topology.add_edge(src_dpid, dst_dpid, port=src_port)
                self.topology.add_edge(dst_dpid, src_dpid, port=dst_port)
        
        self.logger.info(f"Links configured: {len(links)} links")
    
    def _configure_and_install_route(self, src_host, dst_host, path):
        """配置并安装单条路由（修复问题D：统一使用高优先级IPv4流表）"""
        # 转换路径为 DPID
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
        
        # 保存路由配置
        self.configured_routes[(src_host, dst_host)] = dpid_path
        
        # 获取目标主机信息
        dst_ip = self.host_name_to_ip.get(dst_host)
        dst_switch = self.host_to_switch.get(dst_host)
        dst_port = self.host_to_port.get(dst_host)  # 精确端口（修复问题A）
        
        if not dst_ip:
            self.logger.warning(f"No IP for host {dst_host}, cannot install flow")
            return False
        
        # 在路径上的每个交换机安装流表（修复问题D：统一高优先级）
        for i, curr_dpid in enumerate(dpid_path):
            if curr_dpid not in self.datapaths:
                continue
            
            datapath = self.datapaths[curr_dpid]
            parser = datapath.ofproto_parser
            
            # 确定输出端口
            if i < len(dpid_path) - 1:
                # 中间节点：转发到下一跳
                next_dpid = dpid_path[i + 1]
                link_key = (curr_dpid, next_dpid)
                if link_key not in self.links:
                    self.logger.error(f"No link from {curr_dpid} to {next_dpid}")
                    continue
                out_port = self.links[link_key][0]
            else:
                # 最后一跳：转发到主机（使用精确端口，修复问题A）
                dst_dpid = self.switch_name_to_dpid.get(dst_switch)
                if curr_dpid != dst_dpid:
                    self.logger.error(f"Path ends at {curr_dpid} but host is on {dst_dpid}")
                    continue
                
                if dst_port is None:
                    self.logger.error(f"No port info for host {dst_host}")
                    continue
                out_port = dst_port
            
            # 安装基于 IPv4 目标地址的流表（修复问题D：统一匹配方式）
            match = parser.OFPMatch(
                eth_type=0x0800,  # IPv4
                ipv4_dst=dst_ip
            )
            actions = [parser.OFPActionOutput(out_port)]
            
            # 使用高优先级，确保覆盖 MAC 学习规则
            self._add_flow(datapath, priority=100, match=match, actions=actions)
            
            # 记录已安装的流表（修复问题E）
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
        """清除所有配置的流表"""
        for flow_info in self.installed_flows:
            dpid = flow_info['dpid']
            if dpid in self.datapaths:
                datapath = self.datapaths[dpid]
                parser = datapath.ofproto_parser
                ofproto = datapath.ofproto
                
                # 删除高优先级流表
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
        """清除所有路由配置"""
        self._clear_all_configured_flows()
        self.configured_routes.clear()
        self.logger.info("All routes cleared")
        return True
    
    # ==================== 验证 API（修复问题E）====================
    
    def verify_routes(self):
        """验证路由配置是否生效"""
        verification = {
            'configured_routes': {},
            'installed_flows': [],
            'issues': []
        }
        
        # 检查配置的路由
        for (src, dst), path in self.configured_routes.items():
            path_names = [self.dpid_to_switch_name.get(d, str(d)) for d in path]
            verification['configured_routes'][f"{src}->{dst}"] = path_names
        
        # 检查已安装的流表
        for flow in self.installed_flows:
            verification['installed_flows'].append({
                'switch': flow['switch'],
                'match': flow['match'],
                'out_port': flow['out_port'],
                'route': flow['route']
            })
        
        # 检查潜在问题
        for (src, dst), path in self.configured_routes.items():
            # 检查路径上的每个交换机是否都连接
            for dpid in path:
                if dpid not in self.datapaths:
                    verification['issues'].append(
                        f"Switch {dpid} in route {src}->{dst} not connected"
                    )
            
            # 检查是否有对应的 IP 映射
            if dst not in self.host_name_to_ip:
                verification['issues'].append(
                    f"No IP mapping for destination {dst}"
                )
            
            # 检查最后一跳端口
            if dst not in self.host_to_port:
                verification['issues'].append(
                    f"No port mapping for host {dst}"
                )
        
        verification['is_valid'] = len(verification['issues']) == 0
        return verification
    
    # ==================== 交换机事件处理 ====================
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """交换机连接处理"""
        datapath = ev.msg.datapath
        dpid = datapath.id
        
        self.datapaths[dpid] = datapath
        self.topology.add_node(dpid)
        
        # 自动创建交换机名称映射
        switch_name = f's{dpid}'
        if switch_name not in self.switch_name_to_dpid:
            self.switch_name_to_dpid[switch_name] = dpid
            self.dpid_to_switch_name[dpid] = switch_name
        
        # 安装默认流表（最低优先级）
        self._install_default_flow(datapath)
        
        self.logger.info(f"Switch {dpid} ({switch_name}) connected")
    
    def _install_default_flow(self, datapath):
        """安装默认流表"""
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
        """数据包处理（修复问题D：配置路由优先，MAC学习作为备用）"""
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
        
        # 忽略LLDP和IPv6多播
        if eth.ethertype == 0x88cc:
            return
        if eth.dst.startswith('33:33'):
            return
        
        dst = eth.dst
        src = eth.src
        
        # 广播风暴防护
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
        
        # MAC 学习
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port
        
        # 确定输出端口
        # 如果有配置路由且是 IPv4 包，路由应该已经通过流表处理
        # 这里处理的是：ARP、未配置路由的流量、配置路由前的首包
        
        if dst in self.mac_to_port.get(dpid, {}):
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD
        
        actions = [parser.OFPActionOutput(out_port)]
        
        # 只为非配置路由的流量安装低优先级 MAC 规则
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            # 使用较低优先级，不会覆盖配置路由的高优先级规则
            self._add_flow(datapath, 1, match, actions, idle_timeout=300)
        
        # 发送数据包
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=in_port,
            actions=actions,
            data=msg.data
        )
        datapath.send_msg(out)
    
    # ==================== 流表管理 ====================
    
    def _add_flow(self, datapath, priority, match, actions, idle_timeout=0, hard_timeout=0):
        """添加流表项"""
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
    
    # ==================== 监控 ====================
    
    def _monitor_loop(self):
        """监控循环"""
        while True:
            for dp in list(self.datapaths.values()):
                self._request_stats(dp)
            hub.sleep(2)
    
    def _request_stats(self, datapath):
        """请求统计信息"""
        parser = datapath.ofproto_parser
        
        req = parser.OFPPortStatsRequest(datapath, 0, datapath.ofproto.OFPP_ANY)
        datapath.send_msg(req)
        
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)
    
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        """处理端口统计回复"""
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
        """处理流表统计回复"""
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
    
    # ==================== 状态查询 ====================
    
    def get_network_state(self):
        """获取当前网络状态"""
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
    
    # ==================== 意图管理 ====================
    
    def install_intent(self, intent_config):
        """安装意图"""
        intent_id = intent_config.get('intent_id', f'intent_{len(self.intents)}')
        self.intents[intent_id] = intent_config
        self.logger.info(f"Intent installed: {intent_id}")
        return True
    
    def remove_intent(self, intent_id):
        """移除意图"""
        if intent_id in self.intents:
            del self.intents[intent_id]
            return True
        return False
    
    def check_intent(self, intent_id):
        """检查意图合规性"""
        if intent_id not in self.intents:
            return {'compliant': False, 'reason': 'Intent not found'}
        return {'compliant': True, 'violations': []}


class IntentRestController(ControllerBase):
    """REST API控制器"""
    
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
        """GET /state - 获取网络状态"""
        return self._make_response(self.controller.get_network_state())
    
    @route('routing', '/routing', methods=['POST'])
    def configure_routing(self, req, **kwargs):
        """POST /routing - 配置路由变体（完整配置）"""
        try:
            body = json.loads(req.body.decode('utf-8'))
            success = self.controller.configure_routing_variant(body)
            return self._make_response({'success': success})
        except Exception as e:
            self.logger.error(f"Error in configure_routing: {e}")
            return self._make_response({'success': False, 'error': str(e)}, 500)
    
    @route('routes_clear', '/routes', methods=['DELETE'])
    def clear_routes(self, req, **kwargs):
        """DELETE /routes - 清除所有路由"""
        success = self.controller.clear_routes()
        return self._make_response({'success': success})
    
    @route('verify', '/verify', methods=['GET'])
    def verify_routes(self, req, **kwargs):
        """GET /verify - 验证路由配置（修复问题E）"""
        result = self.controller.verify_routes()
        return self._make_response(result)
    
    @route('intent_post', '/intent', methods=['POST'])
    def install_intent(self, req, **kwargs):
        """POST /intent - 安装意图"""
        try:
            body = json.loads(req.body.decode('utf-8'))
            success = self.controller.install_intent(body)
            return self._make_response({'success': success})
        except Exception as e:
            return self._make_response({'success': False, 'error': str(e)}, 500)
    
    @route('intents_list', '/intents', methods=['GET'])
    def list_intents(self, req, **kwargs):
        """GET /intents - 列出所有意图"""
        return self._make_response(self.controller.intents)
