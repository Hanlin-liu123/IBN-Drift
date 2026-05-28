# utils/real_traffic_replay.py
import os
import time
import json
import numpy as np
import threading

class RealTrafficReplayer:
    """真实流量回放器"""
    
    def __init__(self, network_env, traffic_profile_path=None):
        self.network_env = network_env
        self.traffic_profile = None
        self.replay_threads = []
        self.is_replaying = False
        
        if traffic_profile_path:
            self.load_profile(traffic_profile_path)
    
    def load_profile(self, profile_path):
        """加载流量配置文件"""
        with open(profile_path, 'r') as f:
            self.traffic_profile = json.load(f)
        
        print(f"Loaded traffic profile: {self.traffic_profile.get('type', 'unknown')}")
        print(f"  Packets: {len(self.traffic_profile.get('packet_sizes', []))}")
    
    def replay_on_path(self, src_host, dst_host, 
                       duration=60, 
                       rate_scale=1.0,
                       traffic_matrix_value=None):
        """
        在指定路径上回放流量
        
        Args:
            src_host: 源主机
            dst_host: 目的主机
            duration: 持续时间
            rate_scale: 速率缩放因子
            traffic_matrix_value: 流量矩阵中的值（用于缩放）
        """
        if not self.traffic_profile:
            print("No traffic profile loaded")
            return
        
        packet_sizes = self.traffic_profile.get('packet_sizes', [1000])
        inter_arrivals = self.traffic_profile.get('inter_arrival_times', [0.001])
        
        # 根据流量矩阵值缩放速率
        if traffic_matrix_value:
            # 计算需要的包速率来达到目标带宽
            avg_size = np.mean(packet_sizes)
            target_rate_bps = traffic_matrix_value * 1000  # Kbps -> bps
            packets_per_sec = target_rate_bps / (avg_size * 8)
            mean_iat = 1.0 / packets_per_sec if packets_per_sec > 0 else 0.1
            inter_arrivals = [mean_iat * np.random.exponential(1.0) for _ in range(len(packet_sizes))]
        
        # 应用速率缩放
        inter_arrivals = [iat / rate_scale for iat in inter_arrivals]
        
        # 启动回放线程
        thread = threading.Thread(
            target=self._replay_thread,
            args=(src_host, dst_host, packet_sizes, inter_arrivals, duration)
        )
        thread.daemon = True
        thread.start()
        self.replay_threads.append(thread)
    
    def _replay_thread(self, src, dst, packet_sizes, inter_arrivals, duration):
        """回放线程"""
        dst_ip = dst.IP()
        start_time = time.time()
        pkt_idx = 0
        num_packets = len(packet_sizes)
        num_iats = len(inter_arrivals)
        
        while time.time() - start_time < duration and self.is_replaying:
            # 发送数据包
            size = packet_sizes[pkt_idx % num_packets]
            
            # 使用hping3发送指定大小的包
            src.cmd(f'hping3 -c 1 -d {size} --udp -p 12345 {dst_ip} 2>/dev/null &')
            
            # 等待下一个包间隔
            iat = inter_arrivals[pkt_idx % num_iats]
            time.sleep(max(0.0001, iat))  # 最小间隔0.1ms
            
            pkt_idx += 1
    
    def replay_traffic_matrix(self, traffic_matrix, duration=60, rate_scale=1.0):
        """
        回放整个流量矩阵
        
        Args:
            traffic_matrix: numpy数组，行列对应主机
            duration: 持续时间
            rate_scale: 速率缩放
        """
        net = self.network_env.net
        hosts = net.hosts
        n = len(hosts)
        
        if traffic_matrix.shape[0] != n or traffic_matrix.shape[1] != n:
            print(f"Warning: Traffic matrix shape {traffic_matrix.shape} doesn't match {n} hosts")
            # 只使用部分
            n = min(n, traffic_matrix.shape[0], traffic_matrix.shape[1])
        
        self.is_replaying = True
        
        for i in range(n):
            for j in range(n):
                if i != j and traffic_matrix[i, j] > 0:
                    src_host = hosts[i]
                    dst_host = hosts[j]
                    
                    self.replay_on_path(
                        src_host, dst_host,
                        duration=duration,
                        rate_scale=rate_scale,
                        traffic_matrix_value=traffic_matrix[i, j]
                    )
        
        print(f"Started traffic replay with {len(self.replay_threads)} flows")
    
    def replay_with_iperf(self, traffic_matrix, duration=60):
        """
        使用iperf回放流量矩阵（异步方式，不阻塞主线程）

        修复说明：
        原来的实现在主线程里同步执行 dst.cmd() + time.sleep(0.05) + src.cmd()，
        对 22x22=484 个流，总耗时约 484*0.05=24s 以上，导致后续的
        time.sleep(normal_duration) 被完全耗尽，采集线程来不及采集任何样本。

        修复方案：把启动 iperf 的循环放到一个后台线程里异步执行，
        主线程调用后立即返回，不再阻塞后续的采集逻辑。
        """
        net = self.network_env.net
        hosts = net.hosts
        n = min(len(hosts), traffic_matrix.shape[0])

        self.is_replaying = True
        port_base = 5001

        # 把所有要启动的流提前收集好，避免在线程里访问可能变化的外部状态
        flows = []
        for i in range(n):
            for j in range(n):
                if i != j and traffic_matrix[i, j] > 0:
                    flows.append((
                        hosts[i],
                        hosts[j],
                        float(traffic_matrix[i, j]),
                        port_base + i * n + j
                    ))

        def _start_flows():
            for src, dst, rate_kbps, port in flows:
                if not self.is_replaying:
                    break
                # 启动 iperf 服务器（后台，不阻塞）
                dst.cmd(f'iperf -s -u -p {port} &')
                # 启动 iperf 客户端（后台，不阻塞）
                src.cmd(f'iperf -c {dst.IP()} -u -b {rate_kbps}k -t {duration} -p {port} &')

        # 在后台线程里启动所有流，主线程立即返回
        t = threading.Thread(target=_start_flows, daemon=True)
        t.start()
        # 最多等 30 秒让流量建立，超时后主线程继续（不卡死）
        t.join(timeout=30)

        print(f"Started iperf traffic replay for {n}x{n} matrix")
    
    def stop_replay(self):
        """停止所有流量回放"""
        self.is_replaying = False
        
        net = self.network_env.net
        for host in net.hosts:
            host.cmd('killall iperf hping3 2>/dev/null')
        
        self.replay_threads.clear()
        print("Stopped traffic replay")


class TrafficMatrixScaler:
    """流量矩阵缩放器"""
    
    @staticmethod
    def scale_matrix(matrix, target_max_rate=10.0):
        """
        缩放流量矩阵使其适合仿真环境
        
        Args:
            matrix: 原始流量矩阵
            target_max_rate: 目标最大速率(Mbps)
        
        Returns:
            缩放后的矩阵（单位：Kbps）
        """
        if matrix is None or matrix.size == 0:
            return None
        
        max_val = np.max(matrix)
        if max_val > 0:
            scale_factor = (target_max_rate * 1000) / max_val  # 转为Kbps
            return matrix * scale_factor
        return matrix
    
    @staticmethod
    def normalize_matrix(matrix):
        """归一化流量矩阵到[0,1]"""
        if matrix is None or matrix.size == 0:
            return None
        
        max_val = np.max(matrix)
        if max_val > 0:
            return matrix / max_val
        return matrix
    
    @staticmethod
    def add_random_variation(matrix, variation_ratio=0.1):
        """添加随机变化"""
        if matrix is None:
            return None
        
        noise = np.random.uniform(1 - variation_ratio, 1 + variation_ratio, matrix.shape)
        return matrix * noise
