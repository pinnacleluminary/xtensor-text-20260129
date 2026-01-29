from transformers import GenerationConfig
import datetime
from datetime import timezone
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
import os
from typing import Callable, Optional, Dict
import shutil
import json
from transformers.trainer_utils import is_main_process
import wandb
import torch
from state_manager import get_state, set_state
from time_estimation_utils import TimeEstimator, calculate_dynamic_buffer
from convergence_detector import ConvergenceDetector
from typing import Optional
MAX_TRIES = 9


MIS_MATCH_VOCAB_SIZE_MODELS = [
    'NousResearch/Nous-Capybara-7B-V1',
    'berkeley-nest/Starling-LM-7B-alpha',
    'NousResearch/Hermes-2-Theta-Llama-3-8B',
    'MNC-Jihun/Mistral-7B-AO-u0.5-b2-ver0.4'
]

ERROR_GENERATION_CONFIG_MODELS = [
    "lmsys/vicuna-7b-v1.5", 
    "lmsys/vicuna-13b-v1.5",
    "NousResearch/Nous-Hermes-llama-2-7b", 
    "defog/llama-3-sqlcoder-8b"
]

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

print(f"LOCAL_RANK: {LOCAL_RANK} in customized_trainer.py", flush=True)
    
class CustomEvalSaveCallback(TrainerCallback):
    def __init__(
        self,
        function_when_to_evaluate: Callable,
        submission_dir: str,
        output_dir: str,
        original_model_name: str,
        max_steps: int = -1,
        checking_step: int = 100,
        total_steps_all_epochs: int = -1,
        end_time: str = "",
        checking_mode: str = "none",
        model_size: Optional[int] = None
    ):
        self.function_when_to_evaluate = function_when_to_evaluate
        self.submission_dir = submission_dir
        self.current_best_loss = None
        self.best_checkpoint_info = None
        self.update_best_checkpoint = False
        self.output_dir = output_dir
        self.original_model_name = original_model_name
        self.max_steps = max_steps
        self.has_checkpoint = False
        self.save_only = False
        self.checking_step = checking_step
        self.total_steps_all_epochs = total_steps_all_epochs
        self.checking_mode = checking_mode
        self.end_time = end_time
        self.model_size = model_size
        
        self.time_estimator = TimeEstimator(alpha=0.3)
        self.start_time = None
        self.convergence_detector = ConvergenceDetector(patience=3, min_improvement=0.001)
        
    def compute_loss(self, state: TrainerState, metrics):
        return metrics.get("eval_loss", None)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Custom logic to decide whether to save or evaluate
        # print(f"************* on_step_end: {state.global_step}, check eval", flush=True)
        # TODO: implement the logic to save the model without evaluating if there is no check points --> avoid evaluating takes too much time
        # Check if the checking_step is reached
        # print(f"Checking the model at step: {state.global_step}, checking_step: {self.checking_step}, checking_mode: {self.checking_mode}", flush=True)
        
        if self.start_time is None:
            self.start_time = datetime.datetime.now()
        
        if state.global_step > 0:
            elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
            self.time_estimator.update(state.global_step, elapsed_time)
        
        if state.global_step == self.checking_step and self.checking_mode == "first_time":
            # print(f"Checking the model at step: {state.global_step}", flush=True)
            # check the time so far to estimate the training time in total 
            my_state = get_state()
            start_time_obj = datetime.datetime.strptime(my_state["train"]["start_time"], "%Y-%m-%d %H:%M:%S")
            start_train_time_obj = datetime.datetime.strptime(my_state["train"]["start_train_time"], "%Y-%m-%d %H:%M:%S")
            
            log_content = f"Checking the model at step: {state.global_step}"
            now = datetime.datetime.now()
            preparation_time = (start_train_time_obj - start_time_obj).total_seconds()
            log_content += f"\nPreparation time: {preparation_time}"
            time_so_far = (now - start_time_obj).total_seconds()
            log_content += f"\nTime so far: {time_so_far}"
            
            training_time_elapsed = (now - start_train_time_obj).total_seconds()
            time_for_one_step = training_time_elapsed / self.checking_step
            
            ema_time_per_step = self.time_estimator.get_time_per_step()
            if ema_time_per_step:
                time_for_one_step = ema_time_per_step
                log_content += f"\nTime for one step (EMA): {time_for_one_step:.4f}"
            else:
                log_content += f"\nTime for one step: {time_for_one_step:.4f}"
            
            log_content += f"\nTotal steps all epochs: {self.total_steps_all_epochs}"
            
            mean_remaining, lower_bound, upper_bound = self.time_estimator.predict_remaining_time(
                state.global_step, self.total_steps_all_epochs
            )
            total_remaining_training_time = mean_remaining
            log_content += f"\nTotal remaining training time (EMA): {total_remaining_training_time:.2f}s"
            log_content += f"\nConfidence interval: [{lower_bound:.2f}s, {upper_bound:.2f}s]"
            
            # n * time_so_far + total_remaining_training_time = total_remaining_time
            end_time_obj = datetime.datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")
            total_remaining_time = (end_time_obj - now).total_seconds()
            log_content += f"\nTotal remaining time: {total_remaining_time:.2f}s"
            
            dynamic_buffer = calculate_dynamic_buffer(
                estimated_eval_time=500,
                estimated_save_time=300,
                model_size=self.model_size,
                checkpoint_size_mb=None,
                network_latency=0.0
            )
            log_content += f"\nDynamic buffer: {dynamic_buffer:.2f}s"
            
            # n * time_so_far + (time_so_far + total_remaining_training_time) = total_remaining_time
            max_var_time_sofar = 3 * 60
            n = (total_remaining_time - (time_so_far + total_remaining_training_time + dynamic_buffer)) / (time_so_far + max_var_time_sofar)
            n = int(n)
            my_state["check_details"] = {
                "now": str(now.strftime("%Y-%m-%d %H:%M:%S")),
                "start_time": str(start_time_obj.strftime("%Y-%m-%d %H:%M:%S")),
                "start_train_time": str(start_train_time_obj.strftime("%Y-%m-%d %H:%M:%S")),
                "checking_step": self.checking_step,
                "checking_mode": self.checking_mode,
                "estimation_of_steps": n,
                "preparation_time": preparation_time,
                "time_so_far": time_so_far,
                "time_for_one_step": time_for_one_step,
                "ema_time_per_step": ema_time_per_step if ema_time_per_step else time_for_one_step,
                "total_remaining_training_time": total_remaining_training_time,
                "total_remaining_time": total_remaining_time,
                "dynamic_buffer": dynamic_buffer,
                "confidence_interval": {"lower": lower_bound, "upper": upper_bound},
                "end_time": self.end_time,
            }
            if n > 0: # we should try more 
                log_content += f"\nEstimated number of steps to complete the training: {n}"
                control.should_training_stop = True
                control.should_save = False
                args.save_strategy = "no"
                # save the current loss of this step to the state;
                last_log = state.log_history[-1]
                my_state["train"]["current_loss"] = last_log["loss"]
                my_state["mode"] = "continue"
                if n > MAX_TRIES:
                    n = MAX_TRIES
                log_content += f"\nFinal number: {n + 1}"
                my_state["next_runs"] = n + 1 # including the current run
            else:
                print(f"Time is not enough so we will finish the training", flush=True)
                my_state["mode"] = "finish"
            
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
                print(log_content, flush=True)            
            return control
    
        elif state.global_step == self.checking_step and self.checking_mode == "second_time": # at second time, we don't estimate the training time again, just save the current_loss
            log_content = f"Checking the model at step: {state.global_step} where check_mode=second_time"            
            my_state = get_state()
            current_loss = state.log_history[-1]["loss"]
            my_state["train"]["current_loss"] = current_loss
            
            convergence_info = self.convergence_detector.check_convergence(current_loss)
            if convergence_info['converged']:
                print(f"Convergence detected: {convergence_info['reason']}", flush=True)
                log_content += f"\nConvergence detected: {convergence_info['reason']}"
                
            if "runs" in my_state and len(my_state["runs"]) >= 2:
                should_skip = self.convergence_detector.should_skip_remaining_runs(my_state["runs"])
                if should_skip:
                    print(f"Losses are very similar, skipping remaining runs to save time", flush=True)
                    log_content += f"\nSkipping remaining runs (loss variance too low)"
                    my_state["mode"] = "finish"
                    control.should_training_stop = False
                    if is_main_process(LOCAL_RANK):
                        set_state(my_state)
                    return control
                
            control.should_training_stop = True

            # Check if current_loss > current min_loss --> do not save to save time and space
            current_is_the_best = False
            current_min_loss = min([run["current_loss"] for run in my_state["runs"]])
            if current_loss <= current_min_loss:
                if len(my_state["runs"]) + 1 == my_state["next_runs"]:
                    print(f"Current loss: {my_state['train']['current_loss']} is the best so far", flush=True)
                    current_is_the_best = True
                    
            if current_is_the_best:
                control.should_training_stop = False
                my_state["mode"] = "finish"
            else:
                control.should_save = False
                args.save_strategy = "no"
            
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
                # print(log_content, flush=True)
        
            
        when_to_eval = self.function_when_to_evaluate(state.global_step)
        if when_to_eval["eval"]:
            # do not allow the pod to be stopped by any reason 
                # first check if there is at least one checkpoint or not 
            print(f"Evaluating the model at step: {state.global_step} the reason: {when_to_eval['reason']}", flush=True)
            control.should_evaluate = True
            control.should_save = True
            if when_to_eval["reason"] == "end_time":
                if not self.has_checkpoint: # if there is no checkpoint, we just save the model, do not evaluate
                    print(f"No checkpoint found, just save the model at step: {state.global_step}", flush=True)
                    control.should_evaluate = False
                    self.save_only = True
        return control


    def on_evaluate(
        self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs
    ):
        self.save_only = False
        # Append eval_loss to file
        eval_loss = self.compute_loss(state, metrics)
        if state.global_step < 2:
            return 
        print(f"GO INTO CUSTOMIZED EVALUATE AT STEP: {state.global_step}", flush=True)
        if self.best_checkpoint_info is None or eval_loss < self.best_checkpoint_info["loss"]:
            print(f"Updating the best checkpoint info at step: {state.global_step} with eval_loss: {eval_loss}", flush=True)
            self.best_checkpoint_info = {
                "loss": eval_loss,
                "step": state.global_step
            }
            self.update_best_checkpoint = True
        else:
            if self.best_checkpoint_info is not None:
                print(f" At step: {state.global_step} The eval_loss: {eval_loss} is not smaller than the current best eval_loss: {self.best_checkpoint_info['loss']}, update_best_checkpoint={self.update_best_checkpoint}", flush=True)
            

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        
        if state.global_step == self.max_steps and self.max_steps != -1:
            print(f"Stop training because of max steps: {self.max_steps}", flush=True)
            control.should_training_stop = True
        
        self.has_checkpoint = True
        
        if not is_main_process(LOCAL_RANK): # if not main process, skip this
            return 
            
        if self.save_only: # if only save, do not evaluate 
            print(f"Only save the model at step: {state.global_step}, no evaluation", flush=True)
            current_step = state.global_step
            # Remove existing directory if it exists
            if os.path.exists(self.submission_dir):
                shutil.rmtree(self.submission_dir)
                
            shutil.copytree(
                os.path.join(self.output_dir, f"checkpoint-{current_step}"),
                self.submission_dir
            )
            self.update_best_checkpoint = False
            # add a loss.txt file to the submission directory
            with open(os.path.join(self.submission_dir, "loss.txt"), "w") as f:
                f.write(f"{current_step},no_eval")
            
            # release the flag
            self.save_only = False
            return 
            
        # Custom logic after model is saved
        # You can trigger external services, logs, or backups here
        if (
            self.update_best_checkpoint
            and is_main_process(LOCAL_RANK)
        ):
            print(f"Copy the best checkpoint to the submission directory at step: {state.global_step}", flush=True)
            # Remove existing directory if it exists
            if os.path.exists(self.submission_dir):
                shutil.rmtree(self.submission_dir)
            best_eval_loss = self.best_checkpoint_info["loss"]
            shutil.copytree(
                os.path.join(self.output_dir, f"checkpoint-{self.best_checkpoint_info['step']}"),
                self.submission_dir
            )
            self.update_best_checkpoint = False
            # add a loss.txt file to the submission directory
            with open(os.path.join(self.submission_dir, "loss.txt"), "w") as f:
                f.write(f"{self.best_checkpoint_info['step']},{best_eval_loss}")


class GRPOCustomEvalSaveCallback(CustomEvalSaveCallback):
    def compute_loss(self, state: TrainerState, metrics):
        eval_loss = None
        if state.log_history:
            last_log_entry = state.log_history[-1]
            eval_loss = last_log_entry.get("eval_reward", None)
            print(f"choose eval_loss ({eval_loss}) as eval_reward from: last_log_entry: {last_log_entry}; \n metrics: {metrics}", flush=True)
        else:
            print(f"state.log_history is empty", flush=True)
            
        if eval_loss is not None:
            eval_loss = - eval_loss
            
        return eval_loss
    
    def penalize_eval_loss(self, eval_loss: float):
        if eval_loss < 0:
            return eval_loss / 3
        else:
            return eval_loss * 3


def check_remaining_time_less_than_minutes(end_time: str, minutes: int) -> bool: 
    end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    end_time = end_time.replace(tzinfo=timezone.utc)  # Make end_time timezone-aware in UTC
    now = datetime.datetime.now(timezone.utc)
    time_diff = end_time - now
    result =  time_diff.total_seconds() < minutes * 60
    if result:
        print(f"*** current time: {now} end_time: {end_time} time_diff: {time_diff}", flush=True)
    return result


class WhenToEvalHandler:
    def __init__(self, end_time: str, save_before_remaining_time: int = 3, periodic_save_steps: int = -1, steps_per_epoch: int = -1, max_steps: int = -1):
        self.save_before_remaining_time = save_before_remaining_time
        self.run_eval = False
        self.end_time = end_time
        self.periodic_save_steps = periodic_save_steps
        self.steps_per_epoch = steps_per_epoch
        self.max_steps = max_steps

    def __call__(self, global_step: int) -> dict:
        
        if self.steps_per_epoch != -1 and global_step % self.steps_per_epoch == 0 and global_step > 1:
            return {"eval": True, "reason": "epoch"}
        
        if self.periodic_save_steps != -1 and global_step % self.periodic_save_steps == 0 and global_step > 1:
            return {"eval": True, "reason": "periodic"}
        
        if self.save_before_remaining_time > 0 and not self.run_eval:
            if check_remaining_time_less_than_minutes(self.end_time, self.save_before_remaining_time):
                print(f"***ALERT: The time is about to run out need to eval & save the model", flush=True)
                # the eval time might be higher than the end_time, so we need to let the pod not stop by setting a flag for this
                self.run_eval = True
                return {"eval": True, "reason": "end_time"}
        
        if self.max_steps != -1 and global_step == self.max_steps:
            print(f"Stop training because of max steps: {self.max_steps}", flush=True)
            return {"eval": True, "reason": "max_step"}

        return {"eval": False, "reason": "none"}


def set_generation_config(model_name, model):
    try:
        if model_name in ERROR_GENERATION_CONFIG_MODELS:
            model.generation_config = GenerationConfig(temperature=None, top_p=None)
    except:
        print(f"Error setting generation config for model {model_name}")
        pass


def resize_if_needed(model_name, model, token_nums):
    try:
        if model_name in MIS_MATCH_VOCAB_SIZE_MODELS:
            model.resize_token_embeddings(token_nums)
    except:
        print(f"Error resizing token embeddings for model {model_name}")
        pass


def init_wandb(train_request: Dict):
    # set wandb_mode=offline; do not upload the data to wandb export WANDB_MODE=offline
    return True
    task_id = train_request["task_id"]
    expected_repo_name = train_request["expected_repo_name"]
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = train_request["wandb_log_dir"]
    os.environ["WANDB_RUN_ID"] = f"{task_id}_{expected_repo_name}"
    os.environ["WANDB_NAME"] = f"{task_id}_{expected_repo_name}"
    if is_main_process(LOCAL_RANK):
        os.makedirs(train_request["wandb_log_dir"], exist_ok=True)
    return True