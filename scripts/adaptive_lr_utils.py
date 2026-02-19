"""
Adaptive learning rate utilities for time-aware training.
Implements adaptive LR range calculation and intelligent run prioritization.
"""

import os
import json
import numpy as np
from typing import Optional, List, Dict
import lr_utils


def calculate_adaptive_lr_range(
    available_time: float,
    num_runs: int,
    base_lr: float,
    min_range: float = 0.1,
    max_range: float = 0.4,
    task_type: str = "instruct"
) -> float:
    """
    Adjust learning rate exploration range based on available time.
    
    Strategy:
    - More time = wider exploration (can afford to try extreme values)
    - Less time = narrower exploration (focus on promising region)
    - Task-specific defaults
    
    Args:
        available_time: Available time in seconds
        num_runs: Number of runs planned
        base_lr: Base learning rate
        min_range: Minimum log range
        max_range: Maximum log range
        task_type: Task type (instruct, dpo, grpo, chat)
    
    Returns:
        Adaptive log range for learning rate exploration
    """
    # Time per run
    time_per_run = available_time / num_runs if num_runs > 0 else available_time
    
    # Task-specific base ranges
    task_ranges = {
        "instruct": (0.1, 0.28),
        "dpo": (0.1, 0.28),
        "grpo": (0.15, 0.33),
        "chat": (0.1, 0.28)
    }
    
    task_min, task_max = task_ranges.get(task_type.lower(), (min_range, max_range))
    
    # If we have lots of time, explore wider
    if time_per_run > 3600:  # > 1 hour per run
        log_range = task_max
    elif time_per_run > 1800:  # > 30 min per run
        log_range = task_min + (task_max - task_min) * 0.7
    elif time_per_run > 900:  # > 15 min per run
        log_range = task_min + (task_max - task_min) * 0.5
    elif time_per_run > 600:  # > 10 min per run
        log_range = task_min + (task_max - task_min) * 0.3
    else:  # Limited time
        log_range = task_min
    
    return log_range


class IntelligentRunPrioritizer:
    def __init__(self, history_file: Optional[str] = None):
        """
        Initialize intelligent run prioritizer.
        
        Args:
            history_file: Path to history file for persistence
        """
        self.lr_performance_history = {}
        self.history_file = history_file or "/workspace/.lr_performance_history.json"
        self._load_history()
    
    def _load_history(self):
        """Load historical performance data from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.lr_performance_history = json.load(f)
            except:
                self.lr_performance_history = {}
    
    def _save_history(self):
        """Save historical performance data to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.lr_performance_history, f, indent=2)
        except:
            pass
    
    def prioritize_runs(
        self, 
        base_lr: float, 
        num_runs: int, 
        available_time: float,
        model_size: Optional[int] = None,
        log_range: Optional[float] = None
    ) -> List[float]:
        """
        Prioritize learning rates based on multiple factors.
        
        Scoring factors:
        1. Proximity to base_lr (40% weight) - closer = better
        2. Historical performance (30% weight) - better perf = higher priority
        3. Exploration value (20% weight) - unexplored = bonus
        4. Model size adaptation (10% weight) - adjust for model size
        
        Args:
            base_lr: Base learning rate
            num_runs: Number of runs
            available_time: Available time in seconds
            model_size: Model size in parameters
            log_range: Log range for exploration (if None, uses default)
        
        Returns:
            Prioritized list of learning rates
        """
        # Generate candidate learning rates
        if log_range is None:
            log_range = 0.2  # Default
        
        candidates = lr_utils.extend_learning_rates(base_lr, num_runs, log_range=log_range)
        
        # Score candidates
        scored = []
        for lr in candidates:
            score = 0.0
            
            # 1. Proximity score (closer to base = better)
            if max(lr, base_lr) > 0:
                lr_ratio = min(lr, base_lr) / max(lr, base_lr)
            else:
                lr_ratio = 0.5
            proximity_score = lr_ratio * 0.4
            
            # 2. Historical score (if we have data for this model size range)
            hist_score = 0.5 * 0.3  # Default neutral
            if model_size:
                lr_key = f"{lr:.10f}"
                if lr_key in self.lr_performance_history:
                    perf_data = self.lr_performance_history[lr_key]
                    if 'model_size_range' in perf_data:
                        size_range = perf_data['model_size_range']
                        if size_range[0] <= model_size <= size_range[1]:
                            if 'performances' in perf_data and len(perf_data['performances']) > 0:
                                avg_perf = np.mean(perf_data['performances'])
                                hist_score = (1.0 / (1.0 + avg_perf)) * 0.3
            
            # 3. Exploration bonus (unexplored gets bonus)
            lr_key = f"{lr:.10f}"
            exploration_bonus = 0.0
            if lr_key not in self.lr_performance_history:
                exploration_bonus = 0.2 * 0.2
            else:
                # Slightly explored - small bonus
                exploration_bonus = 0.1 * 0.2
            
            # 4. Model size adaptation
            size_score = 0.0
            if model_size:
                # Prefer moderate LRs for large models, wider range for small
                if model_size > 10_000_000_000:  # Large models
                    if 0.8 <= lr / base_lr <= 1.2:  # Close to base
                        size_score = 0.1
                else:  # Small/medium models
                    size_score = 0.05  # Neutral
            
            total_score = proximity_score + hist_score + exploration_bonus + size_score
            scored.append((total_score, lr))
        
        # Sort by score (highest first)
        scored.sort(reverse=True, key=lambda x: x[0])
        
        # Return prioritized list
        prioritized = [lr for _, lr in scored]
        
        # Ensure base_lr is first if it's in the list
        if base_lr in prioritized and prioritized[0] != base_lr:
            prioritized.remove(base_lr)
            prioritized.insert(0, base_lr)
        
        return prioritized
    
    def update_history(
        self, 
        lr: float, 
        performance: float, 
        model_size: Optional[int] = None
    ):
        """
        Update historical performance data.
        
        Args:
            lr: Learning rate used
            performance: Performance metric (lower is better, e.g., loss)
            model_size: Model size in parameters
        """
        lr_key = f"{lr:.10f}"
        if lr_key not in self.lr_performance_history:
            self.lr_performance_history[lr_key] = {
                'performances': [],
                'model_size_range': None
            }
        
        self.lr_performance_history[lr_key]['performances'].append(performance)
        
        # Keep only recent history (last 20 runs)
        if len(self.lr_performance_history[lr_key]['performances']) > 20:
            self.lr_performance_history[lr_key]['performances'] = \
                self.lr_performance_history[lr_key]['performances'][-20:]
        
        # Update model size range
        if model_size:
            current_range = self.lr_performance_history[lr_key]['model_size_range']
            if current_range is None:
                self.lr_performance_history[lr_key]['model_size_range'] = [model_size, model_size]
            else:
                self.lr_performance_history[lr_key]['model_size_range'] = [
                    min(current_range[0], model_size),
                    max(current_range[1], model_size)
                ]
        
        self._save_history()
