# to run:
# diambra run -r "F:\Machine Learning & Data Science\Reinforcement Learning\HuggingFace - DRL Course\Unit Bonus 4 - Diambra\roms" python agent.py --cfgFile "F:\Machine Learning & Data Science\Reinforcement Learning\HuggingFace - DRL Course\Unit Bonus 4 - Diambra\DRL_Advanced_Street-Fighter-III\submission\config.yaml" --trainedModel  "F:\Machine Learning & Data Science\Reinforcement Learning\HuggingFace - DRL Course\Unit Bonus 4 - Diambra\DRL_Advanced_Street-Fighter-III\submission\models\model"

import os
import yaml
import json
import argparse
from diambra.arena import Roles, SpaceTypes, load_settings_flat_dict
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from stable_baselines3 import PPO
import cv2
import numpy as np

def main(cfg_file, trained_model):
    # Read the cfg file
    yaml_file = open(cfg_file)
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    print("Config parameters = ", json.dumps(params, sort_keys=True, indent=4))
    yaml_file.close()

    base_path = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["settings"]["game_id"],
                                params["folders"]["model_name"], "model")

    # Settings
    params["settings"]["action_space"] = SpaceTypes.DISCRETE if params["settings"]["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    settings = load_settings_flat_dict(EnvironmentSettings, params["settings"])
    settings.role = Roles.P1

    # Wrappers Settings
    wrappers_settings = load_settings_flat_dict(WrappersSettings, params["wrappers_settings"])
    wrappers_settings.normalize_reward = False

    # Create environment (avoid vectorized env so will get original env object for rendering)
    env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings, no_vec=True, render_mode="human")
    print("Activated {} environment(s)".format(num_envs))

    # Load the trained agent
    model_path = os.path.join(model_folder, trained_model)
    agent = PPO.load(model_path)

    # Print policy network architecture
    print("Policy architecture:")
    print(agent.policy)

    obs, info = env.reset()

    # Extract the frame shape from the first observation
    frame_shape = obs['frame'].shape
    frame_height, frame_width, frame_channels = frame_shape

    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (frame_width, frame_height))

    while True:    
        env.render()       # to view the environment

        action, _ = agent.predict(obs, deterministic=False)

        obs, reward, terminated, truncated, info = env.step(action.tolist())

        # Access the frame data from the observation dictionary
        frame = obs.get('frame')

        if frame is None:
            print("No frame data found in observation.")
            break

        # Ensure frame is a NumPy array
        frame = np.array(frame)

        # # Check the shape and type of frame
        # print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")

        # Convert frame to the expected format if necessary
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Ensure the frame has three channels (RGB)
        if len(frame.shape) == 2:  # grayscale image
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 1:  # single channel image
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Write the frame to the video file
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if terminated or truncated:
            obs, info = env.reset()
            if info["env_done"]:
                break

    # Close the environment
    env.close()

    # Return success
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgFile", type=str, required=True, help="Configuration file")
    parser.add_argument("--trainedModel", type=str, default="model", help="Model checkpoint")
    opt = parser.parse_args()
    print(opt)

    main(opt.cfgFile, opt.trainedModel)