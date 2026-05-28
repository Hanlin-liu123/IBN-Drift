# data_collection/dataset_formatter.py
"""
数据集格式化器

将采集的网络快照转换为模型训练所需的格式，包括：
- 多粒度特征提取（链路级、路径级、网络级）
- 能耗特征提取
- 标签生成（正常/性能漂移/路径漂移/能耗漂移）
- 数据集划分（训练/验证/测试）
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os


@dataclass
class FeatureConfig:
    """特征配置"""
    # 链路级特征
    link_features: List[str] = None
    
    # 路径级特征
    path_features: List[str] = None
    
    # 网络级特征
    network_features: List[str] = None
    
    # 能耗特征
    energy_features: List[str] = None
    
    # 时序窗口
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
    """
    数据集格式化器
    
    输入：NetworkSnapshot 列表
    输出：用于训练的特征矩阵和标签
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        
        # 特征统计（用于归一化）
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        
    def format_samples(self, samples: List[dict]) -> Dict[str, np.ndarray]:
        """
        格式化样本列表
        
        Args:
            samples: NetworkSnapshot.to_dict() 的列表
            
        Returns:
            {
                'link_features': np.ndarray [N, num_links, link_feat_dim],
                'network_features': np.ndarray [N, network_feat_dim],
                'energy_features': np.ndarray [N, energy_feat_dim],
                'labels': np.ndarray [N],
                'drift_types': List[str]
            }
        """
        if not samples:
            return {}
        
        # 提取特征
        link_features_list = []
        network_features_list = []
        energy_features_list = []
        labels = []
        drift_types = []
        
        for sample in samples:
            # 链路级特征
            link_feat = self._extract_link_features(sample)
            link_features_list.append(link_feat)
            
            # 网络级特征
            net_feat = self._extract_network_features(sample)
            network_features_list.append(net_feat)
            
            # 能耗特征
            energy_feat = self._extract_energy_features(sample)
            energy_features_list.append(energy_feat)
            
            # 标签
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
        """提取链路级特征"""
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
        """提取网络级特征"""
        features = []
        for feat_name in self.config.network_features:
            value = sample.get(feat_name, 0.0)
            features.append(float(value) if value is not None else 0.0)
        
        return np.array(features)
    
    def _extract_energy_features(self, sample: dict) -> np.ndarray:
        """提取能耗特征"""
        energy_data = sample.get('energy', {})
        
        features = []
        for feat_name in self.config.energy_features:
            value = energy_data.get(feat_name, sample.get(feat_name, 0.0))
            features.append(float(value) if value is not None else 0.0)
        
        return np.array(features)
    
    def create_sequences(self, features: np.ndarray, 
                        labels: np.ndarray,
                        sequence_length: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时序序列
        
        Args:
            features: [N, feat_dim]
            labels: [N]
            sequence_length: 序列长度
            
        Returns:
            sequences: [N - seq_len + 1, seq_len, feat_dim]
            seq_labels: [N - seq_len + 1]
        """
        seq_len = sequence_length or self.config.sequence_length
        
        if len(features) < seq_len:
            # 不够长，padding
            padded = np.zeros((seq_len, features.shape[-1]))
            padded[-len(features):] = features
            return padded[np.newaxis, ...], labels[-1:]
        
        sequences = []
        seq_labels = []
        
        for i in range(len(features) - seq_len + 1):
            sequences.append(features[i:i + seq_len])
            seq_labels.append(labels[i + seq_len - 1])  # 使用最后一个时刻的标签
        
        return np.array(sequences), np.array(seq_labels)
    
    def compute_statistics(self, samples: List[dict]):
        """计算特征统计信息"""
        formatted = self.format_samples(samples)
        
        # 网络特征统计
        net_feats = formatted['network_features']
        for i, feat_name in enumerate(self.config.network_features):
            values = net_feats[:, i]
            self.feature_stats[feat_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        # 能耗特征统计
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
        """
        归一化特征
        
        Args:
            features: 特征矩阵
            feature_names: 特征名列表
            method: 'zscore' 或 'minmax'
        """
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
        """
        保存数据集
        
        Args:
            samples: 样本列表
            output_dir: 输出目录
            split_ratios: (train, val, test) 划分比例
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算统计信息
        self.compute_statistics(samples)
        
        # 打乱数据
        indices = np.random.permutation(len(samples))
        
        # 划分数据集
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
        
        # 保存各个split
        for split_name, split_samples in splits.items():
            filepath = os.path.join(output_dir, f'{split_name}.json')
            with open(filepath, 'w') as f:
                json.dump(split_samples, f, indent=2)
            print(f"Saved {len(split_samples)} samples to {filepath}")
        
        # 保存统计信息
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
        """计算标签分布"""
        distribution = {}
        for sample in samples:
            label = sample.get('label', 0)
            drift_type = sample.get('drift_type', 'normal')
            
            if drift_type not in distribution:
                distribution[drift_type] = 0
            distribution[drift_type] += 1
        
        return distribution


class HierarchicalFeatureExtractor:
    """
    层次化特征提取器
    
    对应论文中的多粒度建模：
    - 链路级特征 → Link Encoder
    - 路径级特征 → Path Aggregator  
    - 网络级特征 → Network Encoder
    - 能耗特征 → Energy Encoder (新增)
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
    
    def extract_hierarchical_features(self, sample: dict) -> dict:
        """
        提取层次化特征
        
        Returns:
            {
                'link_level': {link_id: features},
                'path_level': {path_id: features},
                'network_level': features,
                'energy_level': features,
                'cross_level': features  # 跨层次特征
            }
        """
        result = {
            'link_level': {},
            'path_level': {},
            'network_level': None,
            'energy_level': None,
            'cross_level': None
        }
        
        # 链路级
        for link_id, link_data in sample.get('links', {}).items():
            result['link_level'][link_id] = self._extract_link_vector(link_data)
        
        # 路径级
        for path_id, path_data in sample.get('paths', {}).items():
            result['path_level'][path_id] = self._extract_path_vector(path_data)
        
        # 网络级
        result['network_level'] = self._extract_network_vector(sample)
        
        # 能耗级
        result['energy_level'] = self._extract_energy_vector(sample)
        
        # 跨层次特征（关键创新点）
        result['cross_level'] = self._extract_cross_level_features(sample)
        
        return result
    
    def _extract_link_vector(self, link_data: dict) -> np.ndarray:
        """提取链路特征向量"""
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
        """提取路径特征向量"""
        return np.array([
            path_data.get('e2e_delay_ms', 0),
            path_data.get('e2e_jitter_ms', 0),
            path_data.get('e2e_loss_rate', 0),
            path_data.get('e2e_throughput_mbps', 0),
            path_data.get('path_power_watts', 0),
            path_data.get('num_hops', 0)
        ])
    
    def _extract_network_vector(self, sample: dict) -> np.ndarray:
        """提取网络级特征向量"""
        return np.array([
            sample.get('total_throughput_mbps', 0),
            sample.get('avg_delay_ms', 0),
            sample.get('avg_loss_rate', 0),
            sample.get('total_power_watts', 0),
            sample.get('energy_efficiency', 0)
        ])
    
    def _extract_energy_vector(self, sample: dict) -> np.ndarray:
        """提取能耗特征向量"""
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
        """
        提取跨层次特征
        
        这是检测隐蔽能耗漂移的关键：
        - 性能指标和能耗指标的不一致性
        - 设备活跃状态与流量的不匹配
        """
        # 性能-能耗比率特征
        throughput = sample.get('total_throughput_mbps', 1)
        power = sample.get('total_power_watts', 1)
        efficiency = throughput / max(power, 1)
        
        # 活跃设备比率
        energy = sample.get('energy', {})
        total_switches = energy.get('active_switches', 0) + energy.get('sleeping_switches', 1)
        active_ratio = energy.get('active_switches', 0) / max(total_switches, 1)
        
        total_links = energy.get('active_links', 0) + energy.get('sleeping_links', 1)
        link_active_ratio = energy.get('active_links', 0) / max(total_links, 1)
        
        # 延迟-跳数比率（检测次优路由）
        avg_delay = sample.get('avg_delay_ms', 10)
        # 假设正常情况下每跳约5ms
        expected_hops = avg_delay / 5
        actual_active_devices = energy.get('active_switches', 5)
        hop_anomaly = actual_active_devices / max(expected_hops, 1)
        
        return np.array([
            efficiency,
            active_ratio,
            link_active_ratio,
            hop_anomaly,
            throughput / max(active_ratio * 100, 1),  # 每单位活跃度的吞吐
            power / max(throughput, 1)  # 每单位吞吐的功耗
        ])


def create_training_dataset(samples: List[dict], output_dir: str):
    """创建训练数据集的便捷函数"""
    formatter = DatasetFormatter()
    formatter.save_dataset(samples, output_dir)
    return formatter


if __name__ == '__main__':
    # 测试
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
            'total_power_watts': 2000,  # 能耗飙升！
            'energy_efficiency': 0.25,
            'label': 3,  # 能耗漂移
            'drift_type': 'hidden_energy_drift',
            'links': {
                's1-s2': {'delay_ms': 12, 'throughput_mbps': 200, 'power_watts': 100}
            },
            'energy': {
                'total_switch_power': 1600,
                'total_link_power': 400,
                'total_network_power': 2000,
                'active_switches': 7,
                'sleeping_switches': 0  # 全部唤醒
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
    
    # 测试层次化特征提取
    extractor = HierarchicalFeatureExtractor()
    hier_feat = extractor.extract_hierarchical_features(test_samples[1])
    
    print("\nHierarchical features for hidden energy drift sample:")
    print(f"  Network level: {hier_feat['network_level']}")
    print(f"  Energy level: {hier_feat['energy_level']}")
    print(f"  Cross level: {hier_feat['cross_level']}")
