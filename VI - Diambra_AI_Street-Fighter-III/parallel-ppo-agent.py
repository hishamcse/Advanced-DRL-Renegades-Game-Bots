# to run:
# diambra run -r "{roms path}" -s=2 python parallel-ppo-agent.py

from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    # Settings
    settings = EnvironmentSettings()
    # settings.frame_shape = (128, 128, 1)
    settings.characters = ("Ken")

    # Wrappers Settings
    wrappers_settings = WrappersSettings()
    wrappers_settings.normalize_reward = True
    wrappers_settings.stack_frames = 5
    wrappers_settings.add_last_action = True
    wrappers_settings.stack_actions = 12
    wrappers_settings.scale = True
    wrappers_settings.exclude_image_scaling = True
    wrappers_settings.role_relative = True
    wrappers_settings.flatten = True
    wrappers_settings.filter_keys = ["action", "own_health", "opp_health", "own_side", "opp_side",
                                      "opp_character", "stage", "timer"]

    # Create environment
    env, num_envs = make_sb3_env("sfiii3n", settings, wrappers_settings, render_mode="human")
    print("Activated {} environment(s)".format(num_envs))

    agent = PPO("MultiInputPolicy", env, verbose=1)

    # Print policy network architecture
    print("Policy architecture:")
    print(agent.policy)

    print("\nStarting training ...\n")
    agent.learn(total_timesteps=50000, progress_bar=True)
    print("\n .. training completed.")

    # Save the agent
    agent.save("ppo_sfiii3n")
    del agent  # delete trained agent to demonstrate loading

    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the agent was trained vs the current one
    agent = PPO.load("ppo_sfiii3n", env=env, print_system_info=True)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), n_eval_episodes=3)
    print("Reward: {} (avg) ± {} (std)".format(mean_reward, std_reward))

    print("\nStarting trained agent execution ...\n")
    observation = env.reset()
    cumulative_reward = [0.0 for _ in range(num_envs)]
    while True:
        env.render()

        action, _state = agent.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)

        cumulative_reward += reward
        if any(x != 0 for x in reward):
            print("Cumulative reward(s) =", cumulative_reward)

        if done.any():
            observation = env.reset()
            break
    print("\n... trained agent execution completed.\n")

    # Close the environment
    env.close()

    # Return success
    return 0

if __name__ == "__main__":
    main()