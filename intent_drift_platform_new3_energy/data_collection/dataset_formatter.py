# data_collection/dataset_formatter.py

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os


@dataclass
class FeatureConfig:

    # Link-level features
    link_features: List[str] = None
    
    # Path-level features
    path_features: List[str] = None
    
    # Network-level features
    network_features: List[str] = None
    
    # Energy Consumption Characteristics
    energy_features: List[str] = None
    
    # Time window
    sequence_length: int = 10
    
    def __post_init__(self):
        if self.link_features is None:
            self.link_features = [
                'delay_ms', 'jitter_ms', 'loss_rate', 'throughput_mbps',
                'utilization', 'delay_p50', 'delay_p90', 'delay_p99',
                'power_watts'
            ]
        
        if self.path_features is None:
            self.path_features = [
                'e2e_delay_ms', 'e2e_jitter_ms', 'e2e_loss_rate',
                'e2e_throughput_mbps', 'path_power_watts', 'num_hops'
            ]
        
        if self.network_features is None:
            self.network_features = [
                'total_throughput_mbps', 'avg_delay_ms', 'avg_loss_rate',
                'total_power_watts', 'energy_efficiency'
            ]
        
        if self.energy_features is None:
            self.energy_features = [
                'total_switch_power', 'total_link_power', 'total_network_power',
                'active_switches', 'sleeping_switches', 'active_links',
                'sleeping_links', 'energy_efficiency'
            ]


class DatasetFormatter:

    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        
        # Feature statistics (for normalization)
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        
    def format_samples(self, samples: List[dict]) -> Dict[str, np.ndarray]:

        if not samples:
            return {}
        
        # Feature extraction
        link_features_list = []
        network_features_list = []
        energy_features_list = []
        labels = []
        drift_types = []
        
        for sample in samples:
            # Link-level features
            link_feat = self._extract_link_features(sample)
            link_features_list.append(link_feat)
            
            # Network-level features
            net_feat = self._extract_network_features(sample)
            network_features_list.append(net_feat)
            
            # Energy Consumption Characteristics
            energy_feat = self._extract_energy_features(sample)
            energy_features_list.append(energy_feat)
            
            # Labels
            labels.append(sample.get('label', 0))
            drift_types.append(sample.get('drift_type', 'normal'))
        
        return {
            'link_features': np.array(link_features_list),
            'network_features': np.array(network_features_list),
            'energy_features': np.array(energy_features_list),
            'labels': np.array(labels),
            'drift_types': drift_types
        }
    
    def _extract_link_features(self, sample: dict) -> np.ndarray:
        
        links = sample.get('links', {})
        
        if not links:
            return np.zeros((1, len(self.config.link_features)))
        
        features = []
        for link_id, link_data in links.items():
            feat = []
            for feat_name in self.config.link_features:
                value = link_data.get(feat_name, 0.0)
                feat.append(float(value) if value is not None else 0.0)
            features.append(feat)
        
        return np.array(features)
    
    def _extract_network_features(self, sample: dict) -> np.ndarray:
        
        features = []
        for feat_name in self.config.network_features:
            value = sample.get(feat_name, 0.0)
            features.append(float(value) if value is not None else 0.0)
        
        return np.array(features)
    
    def _extract_energy_features(self, sample: dict) -> np.ndarray:
        
        energy_data = sample.get('energy', {})
        
        features = []
        for feat_name in self.config.energy_features:
            value = energy_data.get(feat_name, sample.get(feat_name, 0.0))
            features.append(float(value) if value is not None else 0.0)
        
        return np.array(features)
    
    def create_sequences(self, features: np.ndarray, 
                        labels: np.ndarray,
                        sequence_length: int = None) -> Tuple[np.ndarray, np.ndarray]:

        seq_len = sequence_length or self.config.sequence_length
        
        if len(features) < seq_len:
           
            padded = np.zeros((seq_len, features.shape[-1]))
            padded[-len(features):] = features
            return padded[np.newaxis, ...], labels[-1:]
        
        sequences = []
        seq_labels = []
        
        for i in range(len(features) - seq_len + 1):
            sequences.append(features[i:i + seq_len])
            seq_labels.append(labels[i + seq_len - 1])  # Use the “Last Minute” tag
        
        return np.array(sequences), np.array(seq_labels)
    
    def compute_statistics(self, samples: List[dict]):
        
        formatted = self.format_samples(samples)
        
        # Network Feature Statistics
        net_feats = formatted['network_features']
        for i, feat_name in enumerate(self.config.network_features):
            values = net_feats[:, i]
            self.feature_stats[feat_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        # Statistics on Energy Consumption Characteristics
        energy_feats = formatted['energy_features']
        for i, feat_name in enumerate(self.config.energy_features):
            values = energy_feats[:, i]
            self.feature_stats[f'energy_{feat_name}'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
    
    def normalize(self, features: np.ndarray, 
                 feature_names: List[str],
                 method: str = 'zscore') -> np.ndarray:

        normalized = features.copy()
        
        for i, name in enumerate(feature_names):
            stats = self.feature_stats.get(name, {})
            
            if method == 'zscore':
                mean = stats.get('mean', 0)
                std = stats.get('std', 1)
                if std > 0:
                    normalized[..., i] = (features[..., i] - mean) / std
            
            elif method == 'minmax':
                min_val = stats.get('min', 0)
                max_val = stats.get('max', 1)
                if max_val > min_val:
                    normalized[..., i] = (features[..., i] - min_val) / (max_val - min_val)
        
        return normalized
    
    def save_dataset(self, 
                    samples: List[dict],
                    output_dir: str,
                    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)):

        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate statistical information
        self.compute_statistics(samples)
        
        # Scramble the data
        indices = np.random.permutation(len(samples))
        
        # Split the dataset
        n = len(samples)
        n_train = int(n * split_ratios[0])
        n_val = int(n * split_ratios[1])
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        splits = {
            'train': [samples[i] for i in train_indices],
            'val': [samples[i] for i in val_indices],
            'test': [samples[i] for i in test_indices]
        }
        

        for split_name, split_samples in splits.items():
            filepath = os.path.join(output_dir, f'{split_name}.json')
            with open(filepath, 'w') as f:
                json.dump(split_samples, f, indent=2)
            print(f"Saved {len(split_samples)} samples to {filepath}")
        
        # Save statistics
        stats_path = os.path.join(output_dir, 'stats.json')
        with open(stats_path, 'w') as f:
            json.dump({
                'feature_stats': self.feature_stats,
                'num_samples': {
                    'total': n,
                    'train': len(train_indices),
                    'val': len(val_indices),
                    'test': len(test_indices)
                },
                'label_distribution': self._compute_label_distribution(samples),
                'feature_config': {
                    'link_features': self.config.link_features,
                    'network_features': self.config.network_features,
                    'energy_features': self.config.energy_features
                }
            }, f, indent=2)
        print(f"Saved statistics to {stats_path}")
    
    def _compute_label_distribution(self, samples: List[dict]) -> dict:

        distribution = {}
        for sample in samples:
            label = sample.get('label', 0)
            drift_type = sample.get('drift_type', 'normal')
            
            if drift_type not in distribution:
                distribution[drift_type] = 0
            distribution[drift_type] += 1
        
        return distribution


class HierarchicalFeatureExtractor:

    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
    
    def extract_hierarchical_features(self, sample: dict) -> dict:

        result = {
            'link_level': {},
            'path_level': {},
            'network_level': None,
            'energy_level': None,
            'cross_level': None
        }
        
        # Link-level
        for link_id, link_data in sample.get('links', {}).items():
            result['link_level'][link_id] = self._extract_link_vector(link_data)
        
        # Path-level
        for path_id, path_data in sample.get('paths', {}).items():
            result['path_level'][path_id] = self._extract_path_vector(path_data)
        
        # Network-level
        result['network_level'] = self._extract_network_vector(sample)
        
        # Energy-level
        result['energy_level'] = self._extract_energy_vector(sample)
        
        # Cross-level features (key innovation)
        result['cross_level'] = self._extract_cross_level_features(sample)
        
        return result
    
    def _extract_link_vector(self, link_data: dict) -> np.ndarray:
        """Extract link feature vectors"""
        return np.array([
            link_data.get('delay_ms', 0),
            link_data.get('jitter_ms', 0),
            link_data.get('loss_rate', 0),
            link_data.get('throughput_mbps', 0),
            link_data.get('utilization', 0),
            link_data.get('delay_p50', 0),
            link_data.get('delay_p90', 0),
            link_data.get('delay_p99', 0),
            link_data.get('power_watts', 0)
        ])
    
    def _extract_path_vector(self, path_data: dict) -> np.ndarray:
        """Extract path feature vectors"""
        return np.array([
            path_data.get('e2e_delay_ms', 0),
            path_data.get('e2e_jitter_ms', 0),
            path_data.get('e2e_loss_rate', 0),
            path_data.get('e2e_throughput_mbps', 0),
            path_data.get('path_power_watts', 0),
            path_data.get('num_hops', 0)
        ])
    
    def _extract_network_vector(self, sample: dict) -> np.ndarray:
        """Extract network-level feature vectors"""
        return np.array([
            sample.get('total_throughput_mbps', 0),
            sample.get('avg_delay_ms', 0),
            sample.get('avg_loss_rate', 0),
            sample.get('total_power_watts', 0),
            sample.get('energy_efficiency', 0)
        ])
    
    def _extract_energy_vector(self, sample: dict) -> np.ndarray:
        """Extract energy-level feature vectors"""
        energy = sample.get('energy', {})
        return np.array([
            energy.get('total_switch_power', 0),
            energy.get('total_link_power', 0),
            energy.get('total_network_power', 0),
            energy.get('active_switches', 0),
            energy.get('sleeping_switches', 0),
            energy.get('active_links', 0),
            energy.get('sleeping_links', 0),
            energy.get('energy_efficiency', 0)
        ])
    
    def _extract_cross_level_features(self, sample: dict) -> np.ndarray:

        # Performance-to-Energy-Consumption Ratio Characteristics
        throughput = sample.get('total_throughput_mbps', 1)
        power = sample.get('total_power_watts', 1)
        efficiency = throughput / max(power, 1)
        
        # Active Device Ratio
        energy = sample.get('energy', {})
        total_switches = energy.get('active_switches', 0) + energy.get('sleeping_switches', 1)
        active_ratio = energy.get('active_switches', 0) / max(total_switches, 1)
        
        total_links = energy.get('active_links', 0) + energy.get('sleeping_links', 1)
        link_active_ratio = energy.get('active_links', 0) / max(total_links, 1)
        
        # Delay-Hop Ratio (Detecting Suboptimal Routes)
        avg_delay = sample.get('avg_delay_ms', 10)
        # Assuming that each cycle takes approximately 5 ms under normal conditions
        expected_hops = avg_delay / 5
        actual_active_devices = energy.get('active_switches', 5)
        hop_anomaly = actual_active_devices / max(expected_hops, 1)
        
        return np.array([
            efficiency,
            active_ratio,
            link_active_ratio,
            hop_anomaly,
            throughput / max(active_ratio * 100, 1),  # Throughput per unit of activity
            power / max(throughput, 1)  # Power consumption per unit of throughput
        ])


def create_training_dataset(samples: List[dict], output_dir: str):

    formatter = DatasetFormatter()
    formatter.save_dataset(samples, output_dir)
    return formatter


if __name__ == '__main__':
    
    test_samples = [
        {
            'timestamp': 1.0,
            'total_throughput_mbps': 500,
            'avg_delay_ms': 30,
            'avg_loss_rate': 0.01,
            'total_power_watts': 1000,
            'energy_efficiency': 0.5,
            'label': 0,
            'drift_type': 'normal',
            'links': {
                's1-s2': {'delay_ms': 10, 'throughput_mbps': 200, 'power_watts': 50}
            },
            'energy': {
                'total_switch_power': 800,
                'total_link_power': 200,
                'total_network_power': 1000,
                'active_switches': 5,
                'sleeping_switches': 2
            }
        },
        {
            'timestamp': 2.0,
            'total_throughput_mbps': 500,
            'avg_delay_ms': 35,
            'avg_loss_rate': 0.01,
            'total_power_watts': 2000,  # Soaring energy consumption
            'energy_efficiency': 0.25,
            'label': 3,  # Power consumption drift
            'drift_type': 'hidden_energy_drift',
            'links': {
                's1-s2': {'delay_ms': 12, 'throughput_mbps': 200, 'power_watts': 100}
            },
            'energy': {
                'total_switch_power': 1600,
                'total_link_power': 400,
                'total_network_power': 2000,
                'active_switches': 7,
                'sleeping_switches': 0  # Wake All
            }
        }
    ]
    
    formatter = DatasetFormatter()
    formatted = formatter.format_samples(test_samples)
    
    print("Formatted dataset:")
    print(f"  Network features shape: {formatted['network_features'].shape}")
    print(f"  Energy features shape: {formatted['energy_features'].shape}")
    print(f"  Labels: {formatted['labels']}")
    print(f"  Drift types: {formatted['drift_types']}")
    
    # Testing Hierarchical Feature Extraction
    extractor = HierarchicalFeatureExtractor()
    hier_feat = extractor.extract_hierarchical_features(test_samples[1])
    
    print("\nHierarchical features for hidden energy drift sample:")
    print(f"  Network level: {hier_feat['network_level']}")
    print(f"  Energy level: {hier_feat['energy_level']}")
    print(f"  Cross level: {hier_feat['cross_level']}")
