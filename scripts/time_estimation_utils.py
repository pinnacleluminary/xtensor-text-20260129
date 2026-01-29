import numpy as np
import os
from typing import Optional, Tuple
from model_utility import get_model_num_params


class TimeEstimator:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.ema_time_per_step: Optional[float] = None
        self.step_times = []
        self.variances = []
        self.min_samples = 3
    
    def update(self, step: int, elapsed_time: float):
        if step <= 0:
            return
        
        current_time_per_step = elapsed_time / step
        
        if self.ema_time_per_step is None:
            self.ema_time_per_step = current_time_per_step
        else:
            self.ema_time_per_step = (
                self.alpha * current_time_per_step + 
                (1 - self.alpha) * self.ema_time_per_step
            )
        
        self.step_times.append(current_time_per_step)
        if len(self.step_times) > 10:
            self.step_times = self.step_times[-10:]
            variance = np.var(self.step_times)
            self.variances.append(variance)
            if len(self.variances) > 5:
                self.variances = self.variances[-5:]
    
    def predict_remaining_time(self, current_step: int, total_steps: int) -> Tuple[float, float, float]:
        if self.ema_time_per_step is None:
            return (0.0, 0.0, 0.0)
        
        remaining_steps = total_steps - current_step
        if remaining_steps <= 0:
            return (0.0, 0.0, 0.0)
        
        mean_time = self.ema_time_per_step * remaining_steps
        
        if len(self.variances) >= self.min_samples:
            std_dev = np.sqrt(np.mean(self.variances))
            margin = 1.96 * std_dev * remaining_steps
            lower_bound = max(0, mean_time - margin)
            upper_bound = mean_time + margin
            return (mean_time, lower_bound, upper_bound)
        else:
            return (mean_time, mean_time * 0.8, mean_time * 1.2)
    
    def get_time_per_step(self) -> Optional[float]:
        return self.ema_time_per_step


def calculate_adaptive_checking_step(
    total_steps: int,
    model_size: Optional[int] = None,
    dataset_size: Optional[int] = None,
    min_check_step: int = 50,
    max_check_step: int = 200,
    default_step: int = 100
) -> int:
    if model_size is None and dataset_size is None:
        return default_step
    
    base_step = min_check_step
    
    if model_size:
        if model_size > 10_000_000_000:
            model_factor = 1.5
        elif model_size > 1_000_000_000:
            model_factor = 1.2
        else:
            model_factor = 1.0
    else:
        model_factor = 1.0
    
    if dataset_size:
        dataset_factor = min(1.3, 1.0 + (dataset_size / 1_000_000) * 0.1)
    else:
        dataset_factor = 1.0
    if total_steps > 0:
        steps_factor = min(1.0, total_steps / 1000)
    else:
        steps_factor = 1.0
    
    checking_step = int(base_step * model_factor * dataset_factor * steps_factor)
    return min(max(checking_step, min_check_step), max_check_step)


def calculate_dynamic_buffer(
    estimated_eval_time: Optional[float] = None,
    estimated_save_time: Optional[float] = None,
    model_size: Optional[int] = None,
    checkpoint_size_mb: Optional[float] = None,
    network_latency: float = 0.0,
    min_buffer: float = 300.0
) -> float:
    base_eval_time = estimated_eval_time if estimated_eval_time else 300
    base_save_time = estimated_save_time if estimated_save_time else 180
    
    if model_size:
        size_factor = 1.0 + (model_size / 10_000_000_000) * 0.3
    else:
        size_factor = 1.0

    if checkpoint_size_mb:
        upload_time = checkpoint_size_mb / 10.0
    else:
        upload_time = 0.0
    
    total_ops_time = (base_eval_time + base_save_time) * size_factor
    buffer = (total_ops_time + upload_time + network_latency) * 1.2
    
    return max(buffer, min_buffer)


def get_model_size_from_name(model_name: str, model_path: Optional[str] = None) -> Optional[int]:
    try:
        if model_path:
            return get_model_num_params(model_name, model_path)
        else:
            return get_model_num_params(model_name, model_name)
    except:
        return None


def estimate_dataset_size(dataset_path: Optional[str] = None) -> Optional[int]:
    if not dataset_path or not os.path.exists(dataset_path):
        return None
    
    try:
        import json
        with open(dataset_path, 'r') as f:
            if dataset_path.endswith('.json'):
                data = json.load(f)
                if isinstance(data, list):
                    return len(data)
                elif isinstance(data, dict) and 'data' in data:
                    return len(data['data'])
    except:
        pass
    
    return None
