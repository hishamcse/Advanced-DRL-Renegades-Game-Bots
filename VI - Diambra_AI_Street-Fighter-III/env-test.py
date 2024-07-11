# To run:
# diambra run -r "F:\Machine Learning & Data Science\Reinforcement Learning\HuggingFace - DRL Course\Unit Bonus 4 - Diambra\roms" python basic-env.py

# just to test whether the environment is working or not

import diambra.arena

# Environment creation
env = diambra.arena.make("sfiii3n", render_mode="human")

# Environment reset
observation, info = env.reset(seed=42)

# Agent-Environment interaction loop
while True:
    # (Optional) Environment rendering
    env.render()

    # Action random sampling
    actions = env.action_space.sample()

    # Environment stepping
    observation, reward, terminated, truncated, info = env.step(actions)

    # Episode end (Done condition) check
    if terminated or truncated:
        observation, info = env.reset()
        break

# Environment shutdown
env.close()