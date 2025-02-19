import time
from typing import Dict, Any
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'response_times': [],
            'token_usage': [],
            'costs': [],
            'cache_stats': {'hits': 0, 'misses': 0},
            'errors': [],
            'model_usage': {}
        }
        
    def record_api_call(self, model: str, metrics: Dict[str, Any]):
        """Record metrics for an API call"""
        timestamp = datetime.now().isoformat()
        
        self.metrics['response_times'].append({
            'timestamp': timestamp,
            'model': model,
            'time': metrics['time_taken']
        })
        
        if 'tokens' in metrics:
            self.metrics['token_usage'].append({
                'timestamp': timestamp,
                'model': model,
                'tokens': metrics['tokens']
            })
            
        if 'cost' in metrics:
            self.metrics['costs'].append({
                'timestamp': timestamp,
                'model': model,
                'cost': metrics['cost']
            })
            
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report"""
        return {
            'average_response_time': np.mean([r['time'] for r in self.metrics['response_times']]),
            'total_cost': sum(c['cost'] for c in self.metrics['costs']),
            'cache_efficiency': self.metrics['cache_stats']['hits'] / 
                              (self.metrics['cache_stats']['hits'] + self.metrics['cache_stats']['misses']),
            'model_performance': self._calculate_model_performance()
        }
        
    def _calculate_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics per model"""
        model_metrics = {}
        for model in set(r['model'] for r in self.metrics['response_times']):
            model_times = [r['time'] for r in self.metrics['response_times'] if r['model'] == model]
            model_costs = [c['cost'] for c in self.metrics['costs'] if c['model'] == model]
            
            model_metrics[model] = {
                'avg_response_time': np.mean(model_times),
                'total_cost': sum(model_costs),
                'calls': len(model_times)
            }
            
        return model_metrics 