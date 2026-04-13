import os
from dataclasses import dataclass
from typing import Tuple

import draccus
import tqdm
from libero.libero import benchmark
from libero_env_utils import get_libero_dummy_action, get_libero_env, get_libero_obs, get_max_steps, set_seed_everywhere
from libero_magma_utils import get_magma_action, get_magma_model, get_magma_prompt


@dataclass
class LiberoConfig:
    # Model parameters
    model_name: str = "microsoft/magma-8b-libero-goal"      # model_name
    task_suite_name: str = "libero_goal"                    # Task suite name
    
    # Evaluation parameters
    num_trials_per_task: int = 50                          # Number of rollouts per task
    resolution: int = 256                                  # Image resolution
    num_steps_wait: int = 10                              # Steps to wait for stabilization
    seed: int = 0                                         # Random seed
    save_dir: str = "./libero_eval_log"                   # Directory for saving logs

@draccus.wrap()
def eval_libero(cfg: LiberoConfig) -> Tuple[int, int]:
    """
    Evaluate Libero environment with given configuration.
    
    Args:
        cfg: LiberoConfig object containing evaluation parameters
        
    Returns:
        Tuple[int, int]: Total episodes and total successful episodes
    """
    # Setup logging
    os.makedirs(cfg.save_dir, exist_ok=True)
    log_filepath = f"{cfg.save_dir}/magma_eval-{cfg.task_suite_name}.log"
    log_file = open(log_filepath, "w")
    print(f"Logging to local log file: {log_filepath}")
    
    # Write initial log
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    print(f"Task suite: {cfg.task_suite_name}")

    # Get benchmark and task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    
    # Initialize counters
    total_episodes, total_successes = 0, 0
    set_seed_everywhere(cfg.seed)
    
    # Load model
    processor, magma = get_magma_model(cfg.model_name)
    
    # Iterate through all tasks
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)
        task_name = task.name
        max_steps = get_max_steps(cfg.task_suite_name)
        
        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        
        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, resolution=cfg.resolution)
        print(f"[info] Evaluating task {task_id} from suite {cfg.task_suite_name}, "
              f"the language instruction is {task_description}.")
        log_file.write(f"Task {task_id}: {task_description}\n")
        log_file.flush()

        # Get prompt for current task
        prompt = get_magma_prompt(task_description, processor, magma.config)
        
        # Initialize task-specific counters
        task_episodes, task_successes = 0, 0
        
        # Run trials for current task
        for trial in range(cfg.num_trials_per_task):
            env.reset()
            obs = env.set_init_state(initial_states[trial])

            step = 0
            
            while step < max_steps + cfg.num_steps_wait:
                if step < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action())
                    step += 1
                    continue
                    
                img = get_libero_obs(obs, resize_size=cfg.resolution)
                
                action = get_magma_action(magma, processor, img, prompt, cfg.task_suite_name)
                obs, reward, done, info = env.step(action.tolist())
                step += 1
                
                if done:
                    task_successes += 1
                    break
            
            task_episodes += 1
        
        # Update total counters
        total_episodes += task_episodes
        total_successes += task_successes
        
        # Log task success rate
        task_success_rate = float(task_successes) / float(task_episodes)
        print(f"Current task ({task_name}) success rate: {task_success_rate}")
        log_file.write(f"Current task ({task_name}) success rate: {task_success_rate}\n")
        log_file.flush()
        
    # Log final suite success rate
    suite_success_rate = float(total_successes) / float(total_episodes)
    print(f"Task suite success rate: {suite_success_rate}")
    log_file.write(f"\nTask suite {cfg.task_suite_name} success rate: {suite_success_rate}\n")
    log_file.flush()
    
    env.close()
    log_file.close()
    
    return total_episodes, total_successes

if __name__ == "__main__":
    eval_libero()