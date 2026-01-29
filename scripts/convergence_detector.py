"""
Early convergence detection for time-aware training.
Detects when training has converged and suggests skipping remaining runs.
"""

import numpy as np
from typing import Optional, Dict, List


class ConvergenceDetector:
    def __init__(
        self, 
        patience: int = 3, 
        min_improvement: float = 0.001, 
        convergence_threshold: float = 0.0001
    ):
        """
        Initialize convergence detector.
        
        Args:
            patience: Number of steps without improvement before considering converged
            min_improvement: Minimum loss improvement to count as improvement
            convergence_threshold: Variance threshold for convergence detection
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.convergence_threshold = convergence_threshold
        self.best_loss = float('inf')
        self.no_improvement_count = 0
        self.loss_history = []
    
    def check_convergence(self, current_loss: float) -> Dict:
        """
        Check if training has converged.
        
        Args:
            current_loss: Current training loss
        
        Returns:
            {
                'converged': bool,
                'reason': str,
                'should_skip_remaining': bool
            }
        """
        self.loss_history.append(current_loss)
        
        # Keep only recent history (last 10 steps)
        if len(self.loss_history) > 10:
            self.loss_history = self.loss_history[-10:]
        
        improvement = self.best_loss - current_loss
        
        if improvement > self.min_improvement:
            # Loss is improving
            self.best_loss = current_loss
            self.no_improvement_count = 0
            return {
                'converged': False,
                'reason': 'improving',
                'should_skip_remaining': False
            }
        else:
            # No significant improvement
            self.no_improvement_count += 1
            
            # Check variance-based convergence
            if len(self.loss_history) >= 5:
                loss_variance = np.var(self.loss_history[-5:])
                if loss_variance < self.convergence_threshold:
                    return {
                        'converged': True,
                        'reason': 'low_variance',
                        'should_skip_remaining': True
                    }
            
            # Check patience-based convergence
            if self.no_improvement_count >= self.patience:
                return {
                    'converged': True,
                    'reason': 'no_improvement',
                    'should_skip_remaining': True
                }
            
            return {
                'converged': False,
                'reason': 'waiting',
                'should_skip_remaining': False
            }
    
    def should_skip_remaining_runs(self, runs: List[Dict]) -> bool:
        """
        Determine if remaining runs should be skipped based on current runs.
        
        Strategy:
        - If losses are very similar (< 1% difference), skip remaining
        - If best loss is significantly better, might skip
        - If variance is very low, losses are converging
        
        Args:
            runs: List of run dictionaries with 'current_loss' key
        
        Returns:
            True if remaining runs should be skipped
        """
        if len(runs) < 2:
            return False
        
        losses = [r["current_loss"] for r in runs]
        loss_variance = np.var(losses)
        best_loss = min(losses)
        worst_loss = max(losses)
        
        # If variance is very low, losses are similar - skip remaining
        if loss_variance < 0.0001:
            return True
        
        # If best loss is significantly better (> 5% improvement), might skip
        if worst_loss > 0 and (worst_loss - best_loss) / worst_loss > 0.05:
            # But only if we have enough runs to be confident
            if len(runs) >= 3:
                return True
        
        # If relative difference is very small (< 1%), skip
        if best_loss > 0 and (worst_loss - best_loss) / best_loss < 0.01:
            return True
        
        return False
    
    def reset(self):
        """Reset detector state."""
        self.best_loss = float('inf')
        self.no_improvement_count = 0
        self.loss_history = []
