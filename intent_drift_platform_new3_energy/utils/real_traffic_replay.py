# utils/real_traffic_replay.py
import os
import time
import json
import numpy as np
import threading

class RealTrafficReplayer:
    """Real Traffic Replayer"""
    
    def __init__(self, network_env, traffic_profile_path=None):
        self.network_env = network_env
        self.traffic_profile = None
        self.replay_threads = []
        self.is_replaying = False
        
        if traffic_profile_path:
            self.load_profile(traffic_profile_path)
    
    def load_profile(self, profile_path):
        """Load traffic profile file"""
        with open(profile_path, 'r') as f:
            self.traffic_profile = json.load(f)
        
        print(f"Loaded traffic profile: {self.traffic_profile.get('type', 'unknown')}")
        print(f"  Packets: {len(self.traffic_profile.get('packet_sizes', []))}")
    
    def replay_on_path(self, src_host, dst_host, 
                       duration=60, 
                       rate_scale=1.0,
                       traffic_matrix_value=None):

        if not self.traffic_profile:
            print("No traffic profile loaded")
            return
        
        packet_sizes = self.traffic_profile.get('packet_sizes', [1000])
        inter_arrivals = self.traffic_profile.get('inter_arrival_times', [0.001])
        
        # Scale the rate based on the traffic matrix values
        if traffic_matrix_value:
            # Calculate the required packet rate to achieve the target bandwidth
            avg_size = np.mean(packet_sizes)
            target_rate_bps = traffic_matrix_value * 1000  # Kbps -> bps
            packets_per_sec = target_rate_bps / (avg_size * 8)
            mean_iat = 1.0 / packets_per_sec if packets_per_sec > 0 else 0.1
            inter_arrivals = [mean_iat * np.random.exponential(1.0) for _ in range(len(packet_sizes))]
        
        # Application Rate Scaling
        inter_arrivals = [iat / rate_scale for iat in inter_arrivals]
        
        # Start the playback thread
        thread = threading.Thread(
            target=self._replay_thread,
            args=(src_host, dst_host, packet_sizes, inter_arrivals, duration)
        )
        thread.daemon = True
        thread.start()
        self.replay_threads.append(thread)
    
    def _replay_thread(self, src, dst, packet_sizes, inter_arrivals, duration):
        """Playback thread"""
        dst_ip = dst.IP()
        start_time = time.time()
        pkt_idx = 0
        num_packets = len(packet_sizes)
        num_iats = len(inter_arrivals)
        
        while time.time() - start_time < duration and self.is_replaying:
            # Send a data packet
            size = packet_sizes[pkt_idx % num_packets]
            
            # Send packets of a specified size using hping3
            src.cmd(f'hping3 -c 1 -d {size} --udp -p 12345 {dst_ip} 2>/dev/null &')
            
            # Wait for the next packet
            iat = inter_arrivals[pkt_idx % num_iats]
            time.sleep(max(0.0001, iat))  # Minimum interval: 0.1 ms
            
            pkt_idx += 1
    
    def replay_traffic_matrix(self, traffic_matrix, duration=60, rate_scale=1.0):

        net = self.network_env.net
        hosts = net.hosts
        n = len(hosts)
        
        if traffic_matrix.shape[0] != n or traffic_matrix.shape[1] != n:
            print(f"Warning: Traffic matrix shape {traffic_matrix.shape} doesn't match {n} hosts")
            # Use only a portion
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

        net = self.network_env.net
        hosts = net.hosts
        n = min(len(hosts), traffic_matrix.shape[0])

        self.is_replaying = True
        port_base = 5001

        # Collect all streams to be started in advance to avoid accessing potentially changing external state within the thread.
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
                # Start the iperf server (in the background, non-blocking)
                dst.cmd(f'iperf -s -u -p {port} &')
                # Start the iperf client (in the background, non-blocking)
                src.cmd(f'iperf -c {dst.IP()} -u -b {rate_kbps}k -t {duration} -p {port} &')

        # In the background thread, start all flows, and the main thread returns immediately
        t = threading.Thread(target=_start_flows, daemon=True)
        t.start()
        # Wait up to 30 seconds for the connection to establish; if it times out, the main thread continues (without freezing).
        t.join(timeout=30)

        print(f"Started iperf traffic replay for {n}x{n} matrix")
    
    def stop_replay(self):
        """Stop all traffic playback"""
        self.is_replaying = False
        
        net = self.network_env.net
        for host in net.hosts:
            host.cmd('killall iperf hping3 2>/dev/null')
        
        self.replay_threads.clear()
        print("Stopped traffic replay")


class TrafficMatrixScaler:
    """Traffic Matrix Scaler"""
    
    @staticmethod
    def scale_matrix(matrix, target_max_rate=10.0):

        if matrix is None or matrix.size == 0:
            return None
        
        max_val = np.max(matrix)
        if max_val > 0:
            scale_factor = (target_max_rate * 1000) / max_val  # 转为Kbps
            return matrix * scale_factor
        return matrix
    
    @staticmethod
    def normalize_matrix(matrix):
        """Normalize the traffic matrix to [0,1]"""
        if matrix is None or matrix.size == 0:
            return None
        
        max_val = np.max(matrix)
        if max_val > 0:
            return matrix / max_val
        return matrix
    
    @staticmethod
    def add_random_variation(matrix, variation_ratio=0.1):
        """Add random variation to the traffic matrix"""
        if matrix is None:
            return None
        
        noise = np.random.uniform(1 - variation_ratio, 1 + variation_ratio, matrix.shape)
        return matrix * noise
