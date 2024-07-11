import diambra.arena
from diambra.arena import SpaceTypes, Roles, EnvironmentSettings
import cv2
import numpy as np

# Settings
settings = EnvironmentSettings()
settings.step_ratio = 6
# settings.frame_shape = (512, 512, 1)
settings.role = Roles.P2
settings.difficulty = 4
settings.action_space = SpaceTypes.MULTI_DISCRETE

env = diambra.arena.make("sfiii3n", settings, render_mode="human")
observation, info = env.reset()

# Extract the frame shape from the first observation
frame_shape = observation['frame'].shape
frame_height, frame_width, frame_channels = frame_shape

# Ensure the frame shape matches the expected dimensions for the VideoWriter
if frame_channels != 3:
    raise ValueError("Expected frame to have 3 channels (RGB), but got {frame_channels} channels.")

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width, frame_height))

while True:
    env.render()       # to view the environment

    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action.tolist())
    
    # Access the frame data from the observation dictionary
    frame = observation.get('frame')

    if frame is None:
        print("No frame data found in observation.")
        break

    # Ensure frame is a NumPy array
    frame = np.array(frame)

    # Check the shape and type of frame
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
        break

out.release()

# Close the environment
env.close()