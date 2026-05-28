# utils/mawi_parser.py

import os
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Any

try:
    from scapy.all import PcapReader, IP, TCP, UDP, ICMP
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False
    print("Warning: scapy not installed. Install with: pip install scapy")


class MAWIParser:
    """MAWI Traffic Data Analyzer"""
    
    def __init__(self, pcap_file: str = None):
        self.pcap_file = pcap_file
        self.packet_sizes: List[int] = []
        self.inter_arrival_times: List[float] = []
        self.flow_stats = defaultdict(list)
        self.protocols: Dict[str, int] = defaultdict(int)
    
    def parse_pcap(self, pcap_file: str = None, max_packets: int = 100000) -> Optional[Dict]:

        if not HAS_SCAPY:
            print("Error: scapy required. Install with: pip install scapy")
            return None
        
        pcap_file = pcap_file or self.pcap_file
        if not pcap_file or not os.path.exists(pcap_file):
            print(f"Error: pcap file not found: {pcap_file}")
            return None
        
        print(f"Parsing pcap file: {pcap_file}")
        print(f"Max packets: {max_packets}")
        
        prev_time = None
        packet_count = 0
        
        # Use PcapReader to read packets one by one (to save memory)
        with PcapReader(pcap_file) as pcap_reader:
            for pkt in pcap_reader:
                if packet_count >= max_packets:
                    break
                
                # packet_size
                self.packet_sizes.append(len(pkt))
                
                # inter_arrival_time
                timestamp = float(pkt.time)
                if prev_time is not None:
                    iat = timestamp - prev_time
                    if iat > 0:
                        self.inter_arrival_times.append(iat)
                prev_time = timestamp
                
                # protocol and flow statistics
                if IP in pkt:
                    src = pkt[IP].src
                    dst = pkt[IP].dst
                    
                    if TCP in pkt:
                        self.protocols['TCP'] += 1
                        flow_key = (src, dst, pkt[TCP].sport, pkt[TCP].dport, 'TCP')
                    elif UDP in pkt:
                        self.protocols['UDP'] += 1
                        flow_key = (src, dst, pkt[UDP].sport, pkt[UDP].dport, 'UDP')
                    elif ICMP in pkt:
                        self.protocols['ICMP'] += 1
                        flow_key = (src, dst, 0, 0, 'ICMP')
                    else:
                        self.protocols['OTHER'] += 1
                        flow_key = (src, dst, 0, 0, 'OTHER')
                    
                    self.flow_stats[flow_key].append({
                        'size': len(pkt),
                        'time': timestamp
                    })
                
                packet_count += 1
                if packet_count % 10000 == 0:
                    print(f"  Processed {packet_count} packets...")
        
        print(f"Parsed {len(self.packet_sizes)} packets")
        print(f"Found {len(self.flow_stats)} unique flows")
        
        return self.get_distribution_summary()
    
    def get_distribution_summary(self) -> Optional[Dict]:
        """Get the distribution summary"""
        if not self.packet_sizes:
            return None
        
        sizes = np.array(self.packet_sizes)
        iats = np.array(self.inter_arrival_times) if self.inter_arrival_times else np.array([0.001])
        
        # protocol distribution
        total_packets = sum(self.protocols.values())
        protocol_distribution = {
            proto: count / total_packets 
            for proto, count in self.protocols.items()
        } if total_packets > 0 else {}
        
        return {
            'packet_size': {
                'mean': float(np.mean(sizes)),
                'std': float(np.std(sizes)),
                'min': int(np.min(sizes)),
                'max': int(np.max(sizes)),
                'median': float(np.median(sizes)),
                'percentiles': {
                    'p5': float(np.percentile(sizes, 5)),
                    'p10': float(np.percentile(sizes, 10)),
                    'p25': float(np.percentile(sizes, 25)),
                    'p50': float(np.percentile(sizes, 50)),
                    'p75': float(np.percentile(sizes, 75)),
                    'p90': float(np.percentile(sizes, 90)),
                    'p95': float(np.percentile(sizes, 95)),
                    'p99': float(np.percentile(sizes, 99)),
                },
                'histogram': self._compute_histogram(sizes),
                'common_sizes': self._get_common_sizes(sizes, top_n=10)
            },
            'inter_arrival_time': {
                'mean': float(np.mean(iats)),
                'std': float(np.std(iats)),
                'min': float(np.min(iats)),
                'max': float(np.max(iats)),
                'median': float(np.median(iats)),
                'percentiles': {
                    'p5': float(np.percentile(iats, 5)),
                    'p10': float(np.percentile(iats, 10)),
                    'p25': float(np.percentile(iats, 25)),
                    'p50': float(np.percentile(iats, 50)),
                    'p75': float(np.percentile(iats, 75)),
                    'p90': float(np.percentile(iats, 90)),
                    'p95': float(np.percentile(iats, 95)),
                    'p99': float(np.percentile(iats, 99)),
                }
            },
            'protocol': {
                'counts': dict(self.protocols),
                'distribution': protocol_distribution
            },
            'num_packets': len(self.packet_sizes),
            'num_flows': len(self.flow_stats),
            'source': 'real_pcap'
        }
    
    def _compute_histogram(self, data: np.ndarray, bins: int = 20) -> Dict:
        """Compute the histogram"""
        hist, bin_edges = np.histogram(data, bins=bins)
        return {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }
    
    def _get_common_sizes(self, sizes: np.ndarray, top_n: int = 5) -> List[tuple]:
        """Get the most common packet sizes"""
        unique, counts = np.unique(sizes, return_counts=True)
        sorted_indices = np.argsort(-counts)[:top_n]
        return [(int(unique[i]), int(counts[i])) for i in sorted_indices]
    
    def generate_traffic_profile(self, scale_factor: float = 1.0) -> Optional[Dict]:
        """Generate traffic profile file"""
        if not self.packet_sizes:
            return None
        
        sizes = np.array(self.packet_sizes)
        iats = np.array(self.inter_arrival_times) * scale_factor
        
        profile = {
            'type': 'real_trace',
            'packet_sizes': sizes.tolist()[:10000],
            'inter_arrival_times': iats.tolist()[:10000],
            'scale_factor': scale_factor,
            'size_distribution': {
                'type': 'empirical',
                'mean': float(np.mean(sizes)),
                'std': float(np.std(sizes)),
            },
            'time_distribution': {
                'type': 'empirical',
                'mean': float(np.mean(iats)),
                'std': float(np.std(iats)),
            }
        }
        
        return profile
    
    def save_profile(self, output_path: str, scale_factor: float = 1.0) -> Optional[Dict]:
        """Save the traffic profile file"""
        profile = self.generate_traffic_profile(scale_factor)
        if profile:
            with open(output_path, 'w') as f:
                json.dump(profile, f, indent=2)
            print(f"Saved traffic profile to {output_path}")
        return profile


class SyntheticMAWIProfile:

    
    # Default statistics (used when no real data is available)
    DEFAULT_STATS = {
        'packet_size': {
            'common_sizes': [
                (64, 0.15),
                (576, 0.10),
                (1500, 0.45),
                (500, 0.15),
                (1000, 0.15),
            ],
            'mean': 850,
            'std': 500,
            'min': 64,
            'max': 1500,
            'percentiles': {
                'p10': 64,
                'p50': 1000,
                'p90': 1500,
            }
        },
        'inter_arrival_time': {
            'mean': 0.0001,
            'std': 0.001,
            'min': 0.00001,
            'max': 0.1,
        },
        'protocol': {
            'distribution': {
                'TCP': 0.70,
                'UDP': 0.25,
                'OTHER': 0.05,
            }
        }
    }
    
    def __init__(self, profile_path: str = None):

        self.packet_sizes: List[int] = []
        self.inter_arrival_times: List[float] = []
        self.profile_path = profile_path
        
        if profile_path and os.path.exists(profile_path):
            # Load real-time statistics
            with open(profile_path, 'r') as f:
                self.real_stats = json.load(f)
            self.stats = self._convert_real_stats(self.real_stats)
            self.source = f"Real MAWI statistics from {os.path.basename(profile_path)}"
            print(f"Loaded real MAWI statistics from {profile_path}")
        else:
            # Use default hard-coded features
            self.real_stats = None
            self.stats = self.DEFAULT_STATS
            self.source = "Default synthetic (hardcoded statistics)"
            if profile_path:
                print(f"Warning: Profile not found: {profile_path}, using default statistics")
    
    def _convert_real_stats(self, real_stats: Dict) -> Dict:

        # Extract Package Size Statistics
        pkt_size = real_stats.get('packet_size', {})
        
        # Processing common_sizes: Convert (value, count) to (value, probability)
        common_sizes_raw = pkt_size.get('common_sizes', [])
        if common_sizes_raw:
            # Valid format: [(size, count), ...] or [{‘value’: size, ‘count’: count}, ...]
            if isinstance(common_sizes_raw[0], dict):
                # New Format
                total_count = sum(item['count'] for item in common_sizes_raw)
                common_sizes = [
                    (item['value'], item['count'] / total_count) 
                    for item in common_sizes_raw
                ]
            else:
                # Old format [(size, count), ...]
                total_count = sum(count for _, count in common_sizes_raw)
                common_sizes = [
                    (size, count / total_count) 
                    for size, count in common_sizes_raw
                ]
        else:
            common_sizes = self.DEFAULT_STATS['packet_size']['common_sizes']
        
        # Distribution of Withdrawal Agreements
        protocol_raw = real_stats.get('protocol', {})
        if 'distribution' in protocol_raw:
            protocol_dist = protocol_raw['distribution']
        elif 'counts' in protocol_raw:
            total = sum(protocol_raw['counts'].values())
            protocol_dist = {k: v/total for k, v in protocol_raw['counts'].items()}
        else:
            protocol_dist = self.DEFAULT_STATS['protocol']['distribution']
        
        return {
            'packet_size': {
                'common_sizes': common_sizes,
                'mean': pkt_size.get('mean', self.DEFAULT_STATS['packet_size']['mean']),
                'std': pkt_size.get('std', self.DEFAULT_STATS['packet_size']['std']),
                'min': pkt_size.get('min', self.DEFAULT_STATS['packet_size']['min']),
                'max': pkt_size.get('max', self.DEFAULT_STATS['packet_size']['max']),
                'percentiles': pkt_size.get('percentiles', self.DEFAULT_STATS['packet_size']['percentiles']),
                'histogram': pkt_size.get('histogram', None),
            },
            'inter_arrival_time': {
                'mean': real_stats.get('inter_arrival_time', {}).get('mean', self.DEFAULT_STATS['inter_arrival_time']['mean']),
                'std': real_stats.get('inter_arrival_time', {}).get('std', self.DEFAULT_STATS['inter_arrival_time']['std']),
                'min': real_stats.get('inter_arrival_time', {}).get('min', self.DEFAULT_STATS['inter_arrival_time']['min']),
                'max': real_stats.get('inter_arrival_time', {}).get('max', self.DEFAULT_STATS['inter_arrival_time']['max']),
                'percentiles': real_stats.get('inter_arrival_time', {}).get('percentiles', {}),
            },
            'protocol': {
                'distribution': protocol_dist
            },
            'num_packets': real_stats.get('num_packets', 0),
            'num_flows': real_stats.get('num_flows', 0),
        }
    
    def generate_synthetic_trace(self, num_packets: int = 10000, scale_factor: float = 1.0) -> Dict:

        # ==================== Generated package size ====================
        sizes = []
        common_sizes = self.stats['packet_size']['common_sizes']
        histogram = self.stats['packet_size'].get('histogram')
        
        if histogram and 'counts' in histogram and 'bin_edges' in histogram:
            # Method 1: Histogram-based generation (most realistic)
            counts = np.array(histogram['counts'])
            bin_edges = np.array(histogram['bin_edges'])
            
            # Normalized to a probability
            probs = counts / counts.sum()
            
            for _ in range(num_packets):
                # Select a bin
                bin_idx = np.random.choice(len(probs), p=probs)
                # Take uniform samples within the bin
                low = bin_edges[bin_idx]
                high = bin_edges[bin_idx + 1]
                size = int(np.random.uniform(low, high))
                # Scope of Restrictions
                size = max(self.stats['packet_size']['min'], 
                          min(self.stats['packet_size']['max'], size))
                sizes.append(size)
        
        elif common_sizes:
            # Method 2: Generation Based on Common Value Distributions
            for _ in range(num_packets):
                r = np.random.random()
                cumsum = 0
                selected_size = int(self.stats['packet_size']['mean'])
                
                for size, prob in common_sizes:
                    cumsum += prob
                    if r < cumsum:
                        # Add small perturbations around common values
                        selected_size = size + np.random.randint(-5, 6)
                        break
                
                # Scope of Restrictions
                selected_size = max(self.stats['packet_size']['min'],
                                   min(self.stats['packet_size']['max'], selected_size))
                sizes.append(selected_size)
        
        else:
            # Method 3: Generation Based on a Normal Distribution (Last Resort)
            for _ in range(num_packets):
                size = int(np.random.normal(
                    self.stats['packet_size']['mean'],
                    self.stats['packet_size']['std']
                ))
                size = max(self.stats['packet_size']['min'],
                          min(self.stats['packet_size']['max'], size))
                sizes.append(size)
        
        self.packet_sizes = sizes
        
        # ==================== Generated inter-arrival times ====================
        mean_iat = self.stats['inter_arrival_time']['mean'] * scale_factor
        std_iat = self.stats['inter_arrival_time']['std'] * scale_factor
        
        # Use a Poisson distribution (which better reflects the characteristics of real-world network traffic)
        # using a gamma distribution allows for better control of the variance.
        if std_iat > 0:
            # Using the gamma distribution
            shape = (mean_iat / std_iat) ** 2
            scale = std_iat ** 2 / mean_iat
            iats = np.random.gamma(shape, scale, num_packets - 1)
        else:
            # Using the Exponential Distribution
            iats = np.random.exponential(mean_iat, num_packets - 1)
        
        # Scope of Restrictions
        min_iat = self.stats['inter_arrival_time']['min'] * scale_factor
        max_iat = self.stats['inter_arrival_time']['max'] * scale_factor
        iats = np.clip(iats, min_iat, max_iat)
        
        self.inter_arrival_times = iats.tolist()
        
        # ==================== Return results ====================
        return {
            'packet_sizes': self.packet_sizes,
            'inter_arrival_times': self.inter_arrival_times,
            'scale_factor': scale_factor,
            'source': self.source,
            'statistics_used': {
                'packet_size_mean': self.stats['packet_size']['mean'],
                'packet_size_std': self.stats['packet_size']['std'],
                'iat_mean': self.stats['inter_arrival_time']['mean'],
                'iat_std': self.stats['inter_arrival_time']['std'],
            }
        }
    
    def save_profile(self, output_path: str, num_packets: int = 10000, scale_factor: float = 1.0) -> Dict:

        profile = self.generate_synthetic_trace(num_packets, scale_factor)
        profile['type'] = 'synthetic_mawi'
        
        # Add complete statistics
        profile['full_statistics'] = self.stats
        
        with open(output_path, 'w') as f:
            json.dump(profile, f, indent=2)
        
        print(f"Saved synthetic MAWI profile to {output_path}")
        print(f"  Source: {self.source}")
        print(f"  Packets: {num_packets}")
        print(f"  Scale factor: {scale_factor}")
        
        return profile
    
    def get_statistics_summary(self) -> str:
        """Get a summary of statistics"""
        lines = [
            "=" * 50,
            "MAWI Traffic Statistics Summary",
            "=" * 50,
            f"Source: {self.source}",
            "",
            "[Packet Size]",
            f"  Mean: {self.stats['packet_size']['mean']:.1f} bytes",
            f"  Std: {self.stats['packet_size']['std']:.1f} bytes",
            f"  Min: {self.stats['packet_size']['min']} bytes",
            f"  Max: {self.stats['packet_size']['max']} bytes",
            "",
            "[Inter-Arrival Time]",
            f"  Mean: {self.stats['inter_arrival_time']['mean']*1000:.4f} ms",
            f"  Std: {self.stats['inter_arrival_time']['std']*1000:.4f} ms",
            "",
            "[Protocol Distribution]",
        ]
        
        for proto, ratio in self.stats['protocol']['distribution'].items():
            lines.append(f"  {proto}: {ratio*100:.1f}%")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)


# 便捷函数
def load_mawi_profile(profile_path: str) -> SyntheticMAWIProfile:
    """Load the MAWI statistical feature file"""
    return SyntheticMAWIProfile(profile_path)


def extract_and_save_mawi_stats(pcap_path: str, output_path: str, max_packets: int = 100000) -> Dict:
    """Extract statistical features from a pcap file and save them"""
    parser = MAWIParser(pcap_path)
    stats = parser.parse_pcap(max_packets=max_packets)
    
    if stats:
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to {output_path}")
    
    return stats


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # If a PCAP file is provided, parse it
        pcap_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'mawi_stats.json'
        max_pkts = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
        
        print(f"Extracting statistics from {pcap_file}...")
        stats = extract_and_save_mawi_stats(pcap_file, output_file, max_pkts)
        
        if stats:
            print(f"\nStatistics saved to {output_file}")
            print(f"Packets: {stats['num_packets']}")
            print(f"Flows: {stats['num_flows']}")
    else:
        # Demo using default statistics
        print("Usage: python mawi_parser.py <pcap_file> [output_json] [max_packets]")
        print("\nDemo with default statistics:")
        
        synth = SyntheticMAWIProfile()
        print(synth.get_statistics_summary())
        
        # Generate an example
        trace = synth.generate_synthetic_trace(num_packets=1000)
        print(f"\nGenerated {len(trace['packet_sizes'])} packets")
        print(f"Avg packet size: {np.mean(trace['packet_sizes']):.1f} bytes")
        print(f"Avg IAT: {np.mean(trace['inter_arrival_times'])*1000:.4f} ms")
