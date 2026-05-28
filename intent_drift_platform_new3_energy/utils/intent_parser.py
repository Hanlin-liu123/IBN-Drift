# utils/intent_parser.py
"""
意图解析器

支持解析和验证绿色意图配置，包括：
- 性能约束（QoS）
- 能耗约束（Green）
- 路径约束
- 时间约束
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class IntentType(Enum):
    """意图类型"""
    QOS = "qos"
    GREEN_QOS = "green_qos"
    ENERGY_SAVING = "energy_saving"
    PATH_ENGINEERING = "path_engineering"


class ConstraintOperator(Enum):
    """约束操作符"""
    LE = "<="
    GE = ">="
    EQ = "=="
    LT = "<"
    GT = ">"


@dataclass
class Constraint:
    """单个约束"""
    metric: str
    operator: ConstraintOperator
    threshold: float
    unit: str = ""
    
    def check(self, value: float) -> bool:
        """检查值是否满足约束"""
        if self.operator == ConstraintOperator.LE:
            return value <= self.threshold
        elif self.operator == ConstraintOperator.GE:
            return value >= self.threshold
        elif self.operator == ConstraintOperator.EQ:
            return value == self.threshold
        elif self.operator == ConstraintOperator.LT:
            return value < self.threshold
        elif self.operator == ConstraintOperator.GT:
            return value > self.threshold
        return False


@dataclass
class PerformanceConstraints:
    """性能约束"""
    delay: Optional[Constraint] = None
    loss: Optional[Constraint] = None
    bandwidth: Optional[Constraint] = None
    jitter: Optional[Constraint] = None
    
    def check_all(self, metrics: dict) -> Tuple[bool, List[str]]:
        """检查所有约束"""
        satisfied = True
        violations = []
        
        if self.delay and 'delay_ms' in metrics:
            if not self.delay.check(metrics['delay_ms']):
                satisfied = False
                violations.append(f"delay: {metrics['delay_ms']} > {self.delay.threshold}")
        
        if self.loss and 'loss_rate' in metrics:
            if not self.loss.check(metrics['loss_rate']):
                satisfied = False
                violations.append(f"loss: {metrics['loss_rate']} > {self.loss.threshold}")
        
        if self.bandwidth and 'throughput_mbps' in metrics:
            if not self.bandwidth.check(metrics['throughput_mbps']):
                satisfied = False
                violations.append(f"bandwidth: {metrics['throughput_mbps']} < {self.bandwidth.threshold}")
        
        if self.jitter and 'jitter_ms' in metrics:
            if not self.jitter.check(metrics['jitter_ms']):
                satisfied = False
                violations.append(f"jitter: {metrics['jitter_ms']} > {self.jitter.threshold}")
        
        return satisfied, violations


@dataclass
class EnergyConstraints:
    """能耗约束"""
    max_power: Optional[Constraint] = None
    min_efficiency: Optional[Constraint] = None
    
    # 优化策略
    optimization_mode: str = "balance"
    allow_link_sleep: bool = True
    consolidation_preferred: bool = True
    max_active_devices: int = 100
    
    # 休眠策略
    idle_threshold: float = 0.01
    min_sleep_duration: float = 10.0
    wakeup_delay_tolerance: float = 100.0
    
    def check_all(self, metrics: dict) -> Tuple[bool, List[str]]:
        """检查所有能耗约束"""
        satisfied = True
        violations = []
        
        if self.max_power and 'total_power_watts' in metrics:
            if not self.max_power.check(metrics['total_power_watts']):
                satisfied = False
                violations.append(f"power: {metrics['total_power_watts']} > {self.max_power.threshold}")
        
        if self.min_efficiency and 'energy_efficiency' in metrics:
            if not self.min_efficiency.check(metrics['energy_efficiency']):
                satisfied = False
                violations.append(f"efficiency: {metrics['energy_efficiency']} < {self.min_efficiency.threshold}")
        
        return satisfied, violations


@dataclass
class PathConstraints:
    """路径约束"""
    waypoints: List[str] = field(default_factory=list)
    avoid_nodes: List[str] = field(default_factory=list)
    max_hops: int = 10
    energy_aware_routing: bool = True


@dataclass
class Intent:
    """完整意图"""
    intent_id: str
    name: str
    intent_type: IntentType
    
    # 流量匹配
    match: Dict[str, Any] = field(default_factory=dict)
    
    # 约束
    performance: Optional[PerformanceConstraints] = None
    energy: Optional[EnergyConstraints] = None
    path: Optional[PathConstraints] = None
    
    # 元信息
    priority: int = 100
    description: str = ""
    
    def check_performance(self, metrics: dict) -> Tuple[bool, List[str]]:
        """检查性能约束"""
        if self.performance:
            return self.performance.check_all(metrics)
        return True, []
    
    def check_energy(self, metrics: dict) -> Tuple[bool, List[str]]:
        """检查能耗约束"""
        if self.energy:
            return self.energy.check_all(metrics)
        return True, []
    
    def check_all(self, metrics: dict) -> Dict[str, Any]:
        """
        检查所有约束
        
        返回:
        {
            'performance_satisfied': bool,
            'energy_satisfied': bool,
            'all_satisfied': bool,
            'performance_violations': list,
            'energy_violations': list,
            'drift_type': str  # 'normal', 'performance_drift', 'energy_drift', 'hidden_energy_drift'
        }
        """
        perf_ok, perf_violations = self.check_performance(metrics)
        energy_ok, energy_violations = self.check_energy(metrics)
        
        result = {
            'performance_satisfied': perf_ok,
            'energy_satisfied': energy_ok,
            'all_satisfied': perf_ok and energy_ok,
            'performance_violations': perf_violations,
            'energy_violations': energy_violations,
            'drift_type': 'normal'
        }
        
        # 判断漂移类型
        if not perf_ok and not energy_ok:
            result['drift_type'] = 'mixed_drift'
        elif not perf_ok:
            result['drift_type'] = 'performance_drift'
        elif not energy_ok:
            # 性能达标但能耗超标 = 隐蔽能耗漂移
            result['drift_type'] = 'hidden_energy_drift'
        
        return result


class IntentParser:
    """意图解析器"""
    
    @staticmethod
    def parse_file(filepath: str) -> List[Intent]:
        """解析意图配置文件"""
        with open(filepath, 'r') as f:
            docs = list(yaml.safe_load_all(f))
        
        intents = []
        for doc in docs:
            if doc:
                intent = IntentParser.parse_dict(doc)
                if intent:
                    intents.append(intent)
        
        return intents
    
    @staticmethod
    def parse_dict(data: dict) -> Optional[Intent]:
        """解析意图字典"""
        if not data or 'intent_id' not in data:
            return None
        
        intent_type = IntentType(data.get('type', 'qos'))
        
        # 解析性能约束
        performance = None
        if 'performance' in data:
            perf_data = data['performance']
            performance = PerformanceConstraints(
                delay=IntentParser._parse_constraint(perf_data.get('delay')),
                loss=IntentParser._parse_constraint(perf_data.get('loss')),
                bandwidth=IntentParser._parse_constraint(perf_data.get('bandwidth')),
                jitter=IntentParser._parse_constraint(perf_data.get('jitter'))
            )
        
        # 解析能耗约束
        energy = None
        if 'energy' in data:
            energy_data = data['energy']
            
            optimization = energy_data.get('optimization', {})
            sleep_policy = energy_data.get('sleep_policy', {})
            
            energy = EnergyConstraints(
                max_power=IntentParser._parse_constraint(energy_data.get('max_power')),
                min_efficiency=IntentParser._parse_constraint(energy_data.get('min_efficiency')),
                optimization_mode=optimization.get('mode', 'balance'),
                allow_link_sleep=optimization.get('allow_link_sleep', True),
                consolidation_preferred=optimization.get('consolidation_preferred', True),
                max_active_devices=optimization.get('max_active_devices', 100),
                idle_threshold=sleep_policy.get('idle_threshold', 0.01),
                min_sleep_duration=sleep_policy.get('min_sleep_duration', 10.0),
                wakeup_delay_tolerance=sleep_policy.get('wakeup_delay_tolerance', 100.0)
            )
        
        # 解析路径约束
        path = None
        if 'path' in data:
            path_data = data['path']
            path = PathConstraints(
                waypoints=path_data.get('waypoints', []),
                avoid_nodes=path_data.get('avoid_nodes', []),
                max_hops=path_data.get('max_hops', 10),
                energy_aware_routing=path_data.get('energy_aware_routing', True)
            )
        
        return Intent(
            intent_id=data['intent_id'],
            name=data.get('name', ''),
            intent_type=intent_type,
            match=data.get('match', {}),
            performance=performance,
            energy=energy,
            path=path,
            priority=data.get('priority', 100),
            description=data.get('description', '')
        )
    
    @staticmethod
    def _parse_constraint(data: Optional[dict]) -> Optional[Constraint]:
        """解析单个约束"""
        if not data:
            return None
        
        operator_str = data.get('operator', '<=')
        operator_map = {
            '<=': ConstraintOperator.LE,
            '>=': ConstraintOperator.GE,
            '==': ConstraintOperator.EQ,
            '<': ConstraintOperator.LT,
            '>': ConstraintOperator.GT
        }
        
        return Constraint(
            metric=data.get('metric', ''),
            operator=operator_map.get(operator_str, ConstraintOperator.LE),
            threshold=float(data.get('threshold', 0)),
            unit=data.get('unit', '')
        )


def load_intent(filepath: str) -> Optional[Intent]:
    """加载单个意图"""
    intents = IntentParser.parse_file(filepath)
    return intents[0] if intents else None


def check_intent_satisfaction(intent: Intent, metrics: dict) -> dict:
    """
    检查意图满足情况
    
    这是检测"隐蔽能耗漂移"的核心函数：
    - 如果性能达标但能耗超标，返回 drift_type='hidden_energy_drift'
    """
    return intent.check_all(metrics)


if __name__ == '__main__':
    # 测试意图解析
    import sys
    
    # 创建测试意图
    test_intent = Intent(
        intent_id="test_001",
        name="测试意图",
        intent_type=IntentType.GREEN_QOS,
        performance=PerformanceConstraints(
            delay=Constraint("delay", ConstraintOperator.LE, 50, "ms"),
            loss=Constraint("loss", ConstraintOperator.LE, 0.01, "ratio")
        ),
        energy=EnergyConstraints(
            max_power=Constraint("power", ConstraintOperator.LE, 1500, "watts"),
            min_efficiency=Constraint("efficiency", ConstraintOperator.GE, 2.0, "Mbps/W")
        )
    )
    
    # 测试场景1：正常情况
    print("Test 1: Normal case")
    metrics1 = {
        'delay_ms': 30,
        'loss_rate': 0.005,
        'total_power_watts': 1000,
        'energy_efficiency': 3.0
    }
    result1 = test_intent.check_all(metrics1)
    print(f"  Result: {result1}")
    
    # 测试场景2：性能漂移
    print("\nTest 2: Performance drift")
    metrics2 = {
        'delay_ms': 80,  # 超标
        'loss_rate': 0.005,
        'total_power_watts': 1000,
        'energy_efficiency': 3.0
    }
    result2 = test_intent.check_all(metrics2)
    print(f"  Result: {result2}")
    
    # 测试场景3：隐蔽能耗漂移（重点！）
    print("\nTest 3: Hidden energy drift (performance OK, but energy exceeded)")
    metrics3 = {
        'delay_ms': 30,   # 正常
        'loss_rate': 0.005,  # 正常
        'total_power_watts': 2000,  # 超标！
        'energy_efficiency': 1.0   # 低效！
    }
    result3 = test_intent.check_all(metrics3)
    print(f"  Result: {result3}")
    print(f"  Drift type: {result3['drift_type']}")  # 应该是 hidden_energy_drift
