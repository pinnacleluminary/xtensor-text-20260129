#!/usr/bin/env python3
"""
Demo script to show reg_ratio calculation during training process
This demonstrates what you'll see in the actual training logs.
"""

import math

# Copy of the calculate_reg_ratio function for demo
def calculate_reg_ratio(
    task_type: str = None,
    batch_size: int = None,
    model_params: int = None,
    base_lr: float = None,
    method: str = "experimental"
) -> float:
    """Calculate reg_ratio (learning rate adjustment factor)"""
    if method == "experimental":
        print(f"  [reg_ratio] Using experimental method: returning default value 1.24383", flush=True)
        return 1.24383
    
    elif method == "sqrt_batch":
        if batch_size is None or batch_size <= 0:
            print(f"  [reg_ratio] sqrt_batch method: batch_size={batch_size}, falling back to default 1.24383", flush=True)
            return 1.24383
        reference_batch = 64
        calculated = math.sqrt(batch_size / reference_batch)
        print(f"  [reg_ratio] sqrt_batch method: sqrt({batch_size}/{reference_batch}) = {calculated:.6f}", flush=True)
        return calculated
    
    elif method == "linear_batch":
        if batch_size is None or batch_size <= 0:
            print(f"  [reg_ratio] linear_batch method: batch_size={batch_size}, falling back to default 1.24383", flush=True)
            return 1.24383
        reference_batch = 64
        calculated = batch_size / reference_batch
        print(f"  [reg_ratio] linear_batch method: {batch_size}/{reference_batch} = {calculated:.6f}", flush=True)
        return calculated
    
    elif method == "adaptive":
        reg_ratio = 1.0
        print(f"  [reg_ratio] adaptive method: starting with base=1.0", flush=True)
        
        if batch_size is not None and batch_size > 0:
            reference_batch = 64
            batch_factor = math.sqrt(batch_size / reference_batch)
            print(f"  [reg_ratio]   - batch_size adjustment: sqrt({batch_size}/{reference_batch}) = {batch_factor:.6f}", flush=True)
            reg_ratio *= batch_factor
            print(f"  [reg_ratio]   - after batch adjustment: {reg_ratio:.6f}", flush=True)
        
        if model_params is not None:
            if model_params > 10_000_000_000:
                adjustment = 0.95
                print(f"  [reg_ratio]   - model_size adjustment: {model_params/1e9:.1f}B params -> factor {adjustment:.2f}", flush=True)
                reg_ratio *= adjustment
            elif model_params < 1_000_000_000:
                adjustment = 1.05
                print(f"  [reg_ratio]   - model_size adjustment: {model_params/1e6:.1f}M params -> factor {adjustment:.2f}", flush=True)
                reg_ratio *= adjustment
            else:
                print(f"  [reg_ratio]   - model_size adjustment: {model_params/1e9:.1f}B params -> no adjustment", flush=True)
            print(f"  [reg_ratio]   - after model adjustment: {reg_ratio:.6f}", flush=True)
        
        if task_type:
            task_adjustments = {
                "GrpoTask": 1.0,
                "DpoTask": 1.02,
                "InstructTextTask": 1.02,
                "ChatTask": 1.02,
            }
            task_factor = task_adjustments.get(task_type, 1.0)
            print(f"  [reg_ratio]   - task_type adjustment: {task_type} -> factor {task_factor:.2f}", flush=True)
            reg_ratio *= task_factor
            print(f"  [reg_ratio]   - after task adjustment: {reg_ratio:.6f}", flush=True)
        
        original = reg_ratio
        reg_ratio = max(0.5, min(2.0, reg_ratio))
        if original != reg_ratio:
            print(f"  [reg_ratio]   - clamping: {original:.6f} -> {reg_ratio:.6f} (bounds: 0.5-2.0)", flush=True)
        
        print(f"  [reg_ratio] adaptive method: final result = {reg_ratio:.6f}", flush=True)
        return reg_ratio
    
    else:
        return 1.24383


print("="*70)
print("REG_RATIO CALCULATION DEMONSTRATION")
print("="*70)
print()

# Scenario 1: Experimental method (default) - what happens in normal training
print("\n" + "="*70)
print("SCENARIO 1: Experimental Method (Default - Normal Training)")
print("="*70)
print(f"Calculating reg_ratio using method: 'experimental'")
print(f"Task type: InstructTextTask")
reg_ratio = calculate_reg_ratio(
    task_type="InstructTextTask",
    method="experimental"
)
print(f"\n[OK] Final calculated reg_ratio: {reg_ratio:.6f}")
print(f"\n[TRAINING IMPACT]")
print(f"   Base LR from config: 2.0e-05")
print(f"   After reg_ratio:      {2.0e-05 * reg_ratio:.8f}")
print(f"   Multiplier effect:    {reg_ratio:.2%} increase")

# Scenario 2: Sqrt batch scaling
print("\n" + "="*70)
print("SCENARIO 2: Square Root Batch Scaling")
print("="*70)
print(f"Calculating reg_ratio using method: 'sqrt_batch'")
print(f"Task type: DpoTask")
print(f"Batch size: 128")
reg_ratio = calculate_reg_ratio(
    task_type="DpoTask",
    batch_size=128,
    method="sqrt_batch"
)
print(f"\n[OK] Final calculated reg_ratio: {reg_ratio:.6f}")
print(f"\n[TRAINING IMPACT]")
print(f"   Base LR from config: 2.0e-05")
print(f"   After reg_ratio:      {2.0e-05 * reg_ratio:.8f}")
print(f"   Multiplier effect:    {reg_ratio:.2%} increase")

# Scenario 3: Adaptive method with medium model
print("\n" + "="*70)
print("SCENARIO 3: Adaptive Method (7B Model, 96 batch size)")
print("="*70)
print(f"Calculating reg_ratio using method: 'adaptive'")
print(f"Task type: InstructTextTask")
print(f"Batch size: 96")
print(f"Model params: 7,000,000,000 (7B)")
reg_ratio = calculate_reg_ratio(
    task_type="InstructTextTask",
    batch_size=96,
    model_params=7_000_000_000,
    method="adaptive"
)
print(f"\n[OK] Final calculated reg_ratio: {reg_ratio:.6f}")
print(f"\n[TRAINING IMPACT]")
print(f"   Base LR from config: 2.0e-05")
print(f"   After reg_ratio:      {2.0e-05 * reg_ratio:.8f}")
print(f"   Multiplier effect:    {reg_ratio:.2%} increase")

# Scenario 4: Adaptive with small model
print("\n" + "="*70)
print("SCENARIO 4: Adaptive Method (350M Model - Small Model Bonus)")
print("="*70)
print(f"Calculating reg_ratio using method: 'adaptive'")
print(f"Task type: DpoTask")
print(f"Batch size: 64")
print(f"Model params: 350,000,000 (350M)")
reg_ratio = calculate_reg_ratio(
    task_type="DpoTask",
    batch_size=64,
    model_params=350_000_000,
    method="adaptive"
)
print(f"\n[OK] Final calculated reg_ratio: {reg_ratio:.6f}")
print(f"\n[TRAINING IMPACT]")
print(f"   Base LR from config: 2.0e-05")
print(f"   After reg_ratio:      {2.0e-05 * reg_ratio:.8f}")
print(f"   Multiplier effect:    {reg_ratio:.2%} increase")

print("\n" + "="*70)
print("HOW IT'S USED IN ACTUAL TRAINING")
print("="*70)
print("""
During training, you'll see output like this:

============================================================
REG_RATIO CALCULATION
============================================================
Calculating reg_ratio using method: 'experimental'
Task type: InstructTextTask
  [reg_ratio] Using experimental method: returning default value 1.24383

[OK] Final calculated reg_ratio: 1.243830
============================================================

Then in the config files (instruct_config.py, dpo_config.py, grpo_config.py):
  run_config["learning_rate"] *= train_info["reg_ratio"]

So the learning rate gets multiplied by reg_ratio before training starts.
""")

print("="*70)
