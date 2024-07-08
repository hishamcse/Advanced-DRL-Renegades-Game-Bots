
import cv2
import gym
import numpy as np
import subprocess as sp
# from MAMEToolkit.sf_environment import Environment
# import retro

def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (168, 168))[None, :, :] / 255.0
        return frame
    else:
        return np.zeros((1, 168, 168))

class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", 
                        "{}X{}".format(width, height), "-pix_fmt", "rgb24", "-r", "60", 
                        "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())

class KungFuMasterEnv(object):
    def __init__(self, index, monitor = None):
        self.env = gym.make("ALE/KungFuMaster-v5", render_mode="rgb_array")
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None
        self.env.reset()

    def step(self, action):
        frames, reward, terminated, truncated, info = self.env.step(action)
        game_done = terminated or truncated
        
        if self.monitor:
            self.monitor.record(frames)
            
        frames = process_frame(frames)[None, :, :, :].astype(np.float32)
        
        return frames, reward, game_done

    def reset(self):
        frames, _ = self.env.reset()
        return process_frame(frames)[None, :, :, :].astype(np.float32)

def create_train_env(index, output_path=None):
    num_inputs = 1
    num_actions = gym.make("ALE/KungFuMaster-v5", render_mode="rgb_array").action_space.n
    if output_path:
        monitor = Monitor(160, 210, output_path)
    else:
        monitor = None
    env = KungFuMasterEnv(index, monitor)
    return env, num_inputs, num_actions
