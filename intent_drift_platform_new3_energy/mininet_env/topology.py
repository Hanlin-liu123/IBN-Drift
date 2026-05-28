# mininet_env/topology.py
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.log import setLogLevel
import yaml
import os

class IntentAwareTopo(Topo):
    """Support for intent-aware network topologies"""
    
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        super().__init__()
    
    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def build(self):
        # Use local variables; do not use `self.switches` or `self.hosts`.
        switch_names = {}
        host_names = {}
        
        # Create a switch
        for node in self.config['nodes']:
            if node['type'] == 'switch':
                self.addSwitch(
                    node['id'],
                    protocols='OpenFlow13'
                )
        for host in self.config['hosts']:
            self.addHost(host['id'])
        # Create a host
        for host in self.config['hosts']:
            # Connect the host to the switch
            self.addLink(
                host['id'],
                host['connected_to'],
                cls=TCLink,
                bw=1000  # Host link bandwidth
            )
        
        # Create an inter-switch link
        for link in self.config['links']:
            self.addLink(
                link['src'],
                link['dst'],
                cls=TCLink,
                bw=link['bandwidth'],
                delay=f"{link['delay']}ms"
            )


class NetworkEnvironment:
    """Network simulation environment manager"""
    
    def __init__(self, topo_config, controller_ip='127.0.0.1', controller_port=6653):
        self.topo_config = topo_config
        self.controller_ip = controller_ip
        self.controller_port = controller_port
        self.net = None
        self.topo = None
    
    def start(self):
        """Start the network environment"""
        setLogLevel('info')
        
        # Create Topology
        self.topo = IntentAwareTopo(self.topo_config)
        
        # Create the network
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
        """Stop the network environment"""
        if self.net:
            self.net.stop()
            print("Network stopped")
    
    def get_host(self, name):
        """Get the host object"""
        return self.net.get(name)
    
    def get_switch(self, name):
        """Get the switch object"""
        return self.net.get(name)
    
    def get_link(self, src, dst):
        """Get the link object"""
        return self.net.linksBetween(self.net.get(src), self.net.get(dst))
    
    def modify_link(self, src, dst, **params):
        """Modify link parameters"""
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
