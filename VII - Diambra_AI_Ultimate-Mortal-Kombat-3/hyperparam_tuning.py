# To run:
# diambra run -r "{roms path}" python hyperparam_tuning.py --cfgFile "{config file path .yaml}"

import os
import yaml
import json
import argparse

from typing import Any, Dict
import torch.nn as nn

from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule, AutoSave
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO

import optuna
from optuna.pruners import SuccessiveHalvingPruner, HyperbandPruner, MedianPruner
from optuna.samplers import TPESampler

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparameters.

    :param trial: Optuna trial object
    :return: The sampled hyperparameters for the given trial.
    """
    # Discount factor between 0.9 and 0.9999
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)
    n_epochs = trial.suggest_int("n_epochs", 3, 20)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1, log=True)
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    
    if batch_size > n_steps:
        batch_size = n_steps

    # Display true values
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("n_steps", n_steps)

    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    net_arch = {
        "tiny": [dict(pi=[8, 8], vf=[8, 8])],
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]

    activation_fn = {
        "tanh": nn.Tanh, 
        "relu": nn.ReLU, 
        "elu": nn.ELU, 
        "leaky_relu": nn.LeakyReLU
    }[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        },
    }


def main(cfg_file):
    # Read the cfg file
    yaml_file = open(cfg_file)
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    print("Config parameters = ", json.dumps(params, sort_keys=True, indent=4))
    yaml_file.close()

    base_path = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["settings"]["game_id"],
                                params["folders"]["model_name"], "model")
    tensor_board_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["settings"]["game_id"],
                                        params["folders"]["model_name"], "tb")

    os.makedirs(model_folder, exist_ok=True)

    # Settings
    params["settings"]["action_space"] = SpaceTypes.DISCRETE if params["settings"]["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    settings = load_settings_flat_dict(EnvironmentSettings, params["settings"])

    # Wrappers Settings
    wrappers_settings = load_settings_flat_dict(WrappersSettings, params["wrappers_settings"])

    # Create environment
    env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings, render_mode="human")
    print("Activated {} environment(s)".format(num_envs))

    # Objective function used for optuna study
    def objective(trial):
        
        DEFAULT_HYPERPARAMS = {
            "policy": "MultiInputPolicy",
            "env": env,
        }

        kwargs = DEFAULT_HYPERPARAMS.copy()

        # Sample hyperparameters and update the keyword arguments
        kwargs.update(sample_ppo_params(trial))
        
        # For loop adds option for more robust tuning on multiple seeds if resources are given
        for model_seed in [10]:

            # Create environment
            eval_env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings, render_mode="human")

            # Create the RL model
            model = PPO(**kwargs)

            # Evaluation callback performing a run on the eval_env 
            # (without road boundaries) every 1000 steps
            eval_callback = EvalCallback(
                eval_env=eval_env,
                best_model_save_path=f"/best_model/",
                log_path="./logs/",
                eval_freq=1000,
                deterministic=True,
                render=False,
            )
            
            model.learn(
                int(3e3), 
                callback=[eval_callback],
                progress_bar=True
            )
                        
            env.reset()
            eval_env.reset()
        
        # The metric used for deciding on the best run is the best eval reward recorded at any time
        # -> however the final decision has been made manually taking into account multiple aspects
        # (more is explained in the report)
        return eval_callback.best_mean_reward


    # Creating the study
    study = optuna.create_study(
        sampler=TPESampler(seed=42),  
        pruner=MedianPruner(),
        direction="maximize",
    )

    # Optimizing the study (number of trials not relevant due to resource limitations)
    study.optimize(objective, n_trials=16)

    # Printing the best hyperparameters 
    # (not done due to resource limits being reached before)
    print("Best hyperparameters:", study.best_params)

    # Saving study results into a csv file
    # (not done due to resource limits being reached before)
    df = study.trials_dataframe()
    df.to_csv("optuna_study_results.csv", index=False)
    print("Study results saved to optuna_study_results.csv")

    env.close()

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgFile", type=str, required=True, help="Configuration file")
    opt = parser.parse_args()
    print(opt)

    main(opt.cfgFile)