# utils/qos_config.py
import random

class QoSConfigurator:
    """QoS Configurator - Supports multiple queue scheduling policies"""
    
    # WFQ Weight Configuration (5 Profiles)
    WFQ_PROFILES = [
        {'tos0': 0.90, 'tos1': 0.05, 'tos2': 0.05},
        {'tos0': 0.333, 'tos1': 0.333, 'tos2': 0.334},
        {'tos0': 0.60, 'tos1': 0.30, 'tos2': 0.10},
        {'tos0': 0.50, 'tos1': 0.40, 'tos2': 0.10},
        {'tos0': 0.75, 'tos1': 0.20, 'tos2': 0.05},
    ]
    
    # DRR Weight Allocation
    DRR_PROFILES = [
        {'tos0': 0.80, 'tos1': 0.10, 'tos2': 0.10},
        {'tos0': 0.333, 'tos1': 0.333, 'tos2': 0.334},
        {'tos0': 0.60, 'tos1': 0.30, 'tos2': 0.10},
        {'tos0': 0.70, 'tos1': 0.20, 'tos2': 0.10},
        {'tos0': 0.65, 'tos1': 0.25, 'tos2': 0.10},
    ]
    
    def __init__(self, network_env):
        self.network_env = network_env
        self.current_config = {}
    
    def configure_scenario(self, scenario='scenario1'):
        """Configure QoS Scenario"""
        if scenario == 'scenario1':
            self.configure_scenario1()
        elif scenario == 'scenario2':
            self.configure_scenario2()
        elif scenario == 'scenario3':
            self.configure_scenario3()
        elif scenario == 'scenario4':
            self.configure_scenario4()
        else:
            print(f"Unknown scenario: {scenario}, using FIFO")
    
    def configure_scenario1(self):
        """Scenario 1: WFQ on all nodes, fixed weights 60/30/10"""
        net = self.network_env.net
        profile = self.WFQ_PROFILES[2]  # 60/30/10
        
        for switch in net.switches:
            self._apply_htb_qos(switch, profile)
            self.current_config[switch.name] = {'policy': 'WFQ', 'profile': profile}
        
        print("Applied QoS Scenario 1: All WFQ with 60/30/10 weights")
    
    def configure_scenario2(self):
        """Scenario 2: WFQ on all nodes, random weights"""
        net = self.network_env.net
        
        for switch in net.switches:
            profile = random.choice(self.WFQ_PROFILES)
            self._apply_htb_qos(switch, profile)
            self.current_config[switch.name] = {'policy': 'WFQ', 'profile': profile}
        
        print("Applied QoS Scenario 2: All WFQ with random profiles")
    
    def configure_scenario3(self):
        """Scenario 3: Mixed scheduling policies (SP, WFQ, DRR)"""
        net = self.network_env.net
        
        for switch in net.switches:
            policy = random.choice(['SP', 'WFQ', 'DRR'])
            
            if policy == 'SP':
                self._apply_prio_qos(switch)
                self.current_config[switch.name] = {'policy': 'SP'}
            elif policy == 'WFQ':
                profile = random.choice(self.WFQ_PROFILES)
                self._apply_htb_qos(switch, profile)
                self.current_config[switch.name] = {'policy': 'WFQ', 'profile': profile}
            else:  # DRR
                profile = random.choice(self.DRR_PROFILES)
                self._apply_drr_qos(switch, profile)
                self.current_config[switch.name] = {'policy': 'DRR', 'profile': profile}
        
        print("Applied QoS Scenario 3: Mixed policies (SP/WFQ/DRR)")
    
    def configure_scenario4(self):
        """Scenario 4: Scenario 3 + Uniform ToS Distribution"""
        self.configure_scenario3()
        print("Applied QoS Scenario 4: Mixed policies with uniform ToS")
    
    def _apply_htb_qos(self, switch, profile):
        """Apply HTB (similar to WFQ) Configuration"""
        for intf in switch.intfList():
            if intf.name != 'lo' and not intf.name.startswith('lo'):
                # Remove the existing qdisc
                switch.cmd(f'tc qdisc del dev {intf.name} root 2>/dev/null')
                
                # Add the HTB root qdisc
                switch.cmd(f'tc qdisc add dev {intf.name} root handle 1: htb default 30')
                
                # Add a root class (100 Mbps)
                switch.cmd(f'tc class add dev {intf.name} parent 1: classid 1:1 htb rate 100mbit')
                
                # Add three subclasses
                rate0 = int(100 * profile['tos0'])
                rate1 = int(100 * profile['tos1'])
                rate2 = int(100 * profile['tos2'])
                
                switch.cmd(f'tc class add dev {intf.name} parent 1:1 classid 1:10 htb rate {rate0}mbit')
                switch.cmd(f'tc class add dev {intf.name} parent 1:1 classid 1:20 htb rate {rate1}mbit')
                switch.cmd(f'tc class add dev {intf.name} parent 1:1 classid 1:30 htb rate {rate2}mbit')
                
                # Add a filter (based on DSCP)
                switch.cmd(f'tc filter add dev {intf.name} parent 1: protocol ip prio 1 u32 match ip tos 0xb8 0xff flowid 1:10')
                switch.cmd(f'tc filter add dev {intf.name} parent 1: protocol ip prio 2 u32 match ip tos 0x68 0xff flowid 1:20')
    
    def _apply_prio_qos(self, switch):
        """Apply strict priority configuration"""
        for intf in switch.intfList():
            if intf.name != 'lo' and not intf.name.startswith('lo'):
                switch.cmd(f'tc qdisc del dev {intf.name} root 2>/dev/null')
                switch.cmd(f'tc qdisc add dev {intf.name} root handle 1: prio bands 3')
                
                # Add filters
                switch.cmd(f'tc filter add dev {intf.name} parent 1: protocol ip prio 1 u32 match ip tos 0xb8 0xff flowid 1:1')
                switch.cmd(f'tc filter add dev {intf.name} parent 1: protocol ip prio 2 u32 match ip tos 0x68 0xff flowid 1:2')
    
    def _apply_drr_qos(self, switch, profile):
        """Apply DRR configuration"""
        for intf in switch.intfList():
            if intf.name != 'lo' and not intf.name.startswith('lo'):
                switch.cmd(f'tc qdisc del dev {intf.name} root 2>/dev/null')
                
                # DRR requires a special kernel module; here, we use SFQ as an approximation.
                switch.cmd(f'tc qdisc add dev {intf.name} root handle 1: sfq perturb 10')
    
    def clear_qos(self):
        """Clear all QoS configurations"""
        net = self.network_env.net
        for switch in net.switches:
            for intf in switch.intfList():
                if intf.name != 'lo':
                    switch.cmd(f'tc qdisc del dev {intf.name} root 2>/dev/null')
        
        self.current_config.clear()
        print("Cleared all QoS configurations")
    
    def get_config_summary(self):
        """Get a summary of the current configuration"""
        return self.current_config