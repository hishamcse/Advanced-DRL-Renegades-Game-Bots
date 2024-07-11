# always selects action with index 0 (no action)

import diambra.arena
from diambra.arena import SpaceTypes, Roles, EnvironmentSettings

# Settings
settings = EnvironmentSettings()
settings.step_ratio = 6
settings.frame_shape = (128, 128, 1)
settings.role = Roles.P2
settings.difficulty = 4
settings.action_space = SpaceTypes.MULTI_DISCRETE

env = diambra.arena.make("sfiii3n", settings)
observation, info = env.reset()

while True:
    action = env.get_no_op_action()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

# Close the environment
env.close()