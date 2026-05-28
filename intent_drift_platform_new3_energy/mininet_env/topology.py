# mininet_env/topology.py
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.log import setLogLevel
import yaml
import os

class IntentAwareTopo(Topo):
    """支持意图感知的网络拓扑"""
    
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        super().__init__()
    
    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def build(self):
        # 使用局部变量，不要用self.switches和self.hosts
        switch_names = {}
        host_names = {}
        
        # 创建交换机
        for node in self.config['nodes']:
            if node['type'] == 'switch':
                self.addSwitch(
                    node['id'],
                    protocols='OpenFlow13'
                )
        for host in self.config['hosts']:
            self.addHost(host['id'])
        # 创建主机
        for host in self.config['hosts']:
            # 连接主机到交换机
            self.addLink(
                host['id'],
                host['connected_to'],
                cls=TCLink,
                bw=1000  # 主机链路带宽
            )
        
        # 创建交换机间链路
        for link in self.config['links']:
            self.addLink(
                link['src'],
                link['dst'],
                cls=TCLink,
                bw=link['bandwidth'],
                delay=f"{link['delay']}ms"
            )


class NetworkEnvironment:
    """网络仿真环境管理器"""
    
    def __init__(self, topo_config, controller_ip='127.0.0.1', controller_port=6653):
        self.topo_config = topo_config
        self.controller_ip = controller_ip
        self.controller_port = controller_port
        self.net = None
        self.topo = None
    
    def start(self):
        """启动网络环境"""
        setLogLevel('info')
        
        # 创建拓扑
        self.topo = IntentAwareTopo(self.topo_config)
        
        # 创建网络
        self.net = Mininet(
            topo=self.topo,
            controller=RemoteController(
                'c0',
                ip=self.controller_ip,
                port=self.controller_port
            ),
            switch=OVSKernelSwitch,
            link=TCLink,
            autoSetMacs=True
        )
        
        self.net.start()
        print(f"Network started with {len(self.net.switches)} switches and {len(self.net.hosts)} hosts")
        
        return self.net
    
    def stop(self):
        """停止网络环境"""
        if self.net:
            self.net.stop()
            print("Network stopped")
    
    def get_host(self, name):
        """获取主机对象"""
        return self.net.get(name)
    
    def get_switch(self, name):
        """获取交换机对象"""
        return self.net.get(name)
    
    def get_link(self, src, dst):
        """获取链路对象"""
        return self.net.linksBetween(self.net.get(src), self.net.get(dst))
    
    def modify_link(self, src, dst, **params):
        """修改链路参数"""
        links = self.get_link(src, dst)
        if links:
            link = links[0]
            intf1, intf2 = link.intf1, link.intf2
            
            if 'delay' in params:
                delay = params['delay']
                intf1.config(delay=f"{delay}ms")
                intf2.config(delay=f"{delay}ms")
            
            if 'bw' in params:
                bw = params['bw']
                intf1.config(bw=bw)
                intf2.config(bw=bw)
            
            if 'loss' in params:
                loss = params['loss']
                intf1.config(loss=loss)
                intf2.config(loss=loss)
            
            print(f"Modified link {src}-{dst}: {params}")
