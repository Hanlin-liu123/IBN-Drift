# utils/traffic_generator.py
import time
import random
import numpy as np
import threading
from enum import Enum

class TimeDistribution(Enum):
    EXPONENTIAL = 0  # Poisson process
    UNIFORM = 1
    DETERMINISTIC = 2

class SizeDistribution(Enum):
    DETERMINISTIC = 0
    UNIFORM = 1
    BINOMIAL = 2

class ToS(Enum):
    HIGH_PRIORITY = 0
    MEDIUM_PRIORITY = 1
    LOW_PRIORITY = 2

class TrafficGenerator:
    """流量生成器 - 模仿BNN-UPC的流量模型"""
    
    def __init__(self, network_env):
        self.network_env = network_env
        self.active_flows = []
    
    def generate_traffic_matrix(self, 
                                 sparsity=0.3,
                                 max_rate=10,
                                 tos_distribution=[0.1, 0.3, 0.6]):
        """
        生成流量矩阵
        
        Args:
            sparsity: 有流量的主机对比例
            max_rate: 最大流量速率 (Mbps)
            tos_distribution: [高优先级, 中优先级, 低优先级] 概率
        
        Returns:
            traffic_matrix: {(src, dst): config}
        """
        net = self.network_env.net
        hosts = net.hosts
        traffic_matrix = {}
        
        for src in hosts:
            for dst in hosts:
                if src != dst and random.random() < sparsity:
                    # 流量速率
                    rate = random.uniform(0.1, max_rate)
                    
                    # 选择ToS
                    tos = np.random.choice([0, 1, 2], p=tos_distribution)
                    
                    # 包大小分布（二项式）
                    pkt_size_small = 64   # ACK等小包
                    pkt_size_large = 1500  # 数据大包
                    
                    traffic_matrix[(src.name, dst.name)] = {
                        'rate': rate,
                        'tos': tos,
                        'time_distribution': TimeDistribution.EXPONENTIAL.value,
                        'size_distribution': SizeDistribution.BINOMIAL.value,
                        'pkt_size_small': pkt_size_small,
                        'pkt_size_large': pkt_size_large
                    }
        
        return traffic_matrix
    
    def apply_traffic(self, traffic_matrix, duration=30):
        """应用流量矩阵到Mininet"""
        net = self.network_env.net
        
        for (src_name, dst_name), config in traffic_matrix.items():
            src = net.get(src_name)
            dst = net.get(dst_name)
            
            if src and dst:
                self._start_iperf_flow(
                    src, dst, 
                    rate_mbps=config['rate'],
                    tos=config['tos'],
                    duration=duration
                )
    
    def _start_iperf_flow(self, src, dst, rate_mbps, tos=0, duration=30):
        """启动iperf流量"""
        dst_ip = dst.IP()
        port = 5001 + len(self.active_flows)
        
        # DSCP值映射
        dscp_map = {0: 46, 1: 26, 2: 0}  # EF, AF31, BE
        dscp = dscp_map.get(tos, 0)
        
        # 启动iperf服务器
        dst.popen(['iperf', '-s', '-u', '-p', str(port)])
        time.sleep(0.1)
        
        # 启动iperf客户端
        rate_kbps = int(rate_mbps * 1000)
        src.popen(['iperf', '-c', dst_ip, '-u', '-b', f'{rate_kbps}k', '-t', str(duration), '-p', str(port), '-S', str(dscp)])
        
        self.active_flows.append({
            'src': src.name,
            'dst': dst.name,
            'rate': rate_mbps,
            'tos': tos,
            'port': port
        })
    
    def generate_background_traffic(self, duration=60, intensity='medium'):
        """生成背景流量"""
        intensity_config = {
            'low': {'sparsity': 0.1, 'max_rate': 2},
            'medium': {'sparsity': 0.3, 'max_rate': 5},
            'high': {'sparsity': 0.5, 'max_rate': 10}
        }
        
        config = intensity_config.get(intensity, intensity_config['medium'])
        traffic_matrix = self.generate_traffic_matrix(**config)
        self.apply_traffic(traffic_matrix, duration)
        
        print(f"Started {len(traffic_matrix)} background flows ({intensity} intensity)")
        return traffic_matrix
    
    def stop_all_traffic(self):
        """停止所有流量"""
        net = self.network_env.net
        for host in net.hosts:
            host.popen(['killall', '-9', 'iperf', 'iperf3'])
        self.active_flows.clear()
        print("Stopped all traffic")
