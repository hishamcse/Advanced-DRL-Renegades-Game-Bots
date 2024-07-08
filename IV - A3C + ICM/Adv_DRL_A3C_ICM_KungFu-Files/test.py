import os
os.environ['OMP_NUM_THREADS'] = '1'

import gym
import cv2
import argparse
import torch
from env import create_train_env
from model import ActorCritic
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

def get_eval_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Curiosity-driven Exploration by Self-supervised Prediction for kungfu master""")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output_path", type=str, default="results")
#     args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    return args

def evaluate(opt):
#     torch.manual_seed(123)
    env, num_states, num_actions = create_train_env(0, "{}/video.mp4".format(opt.output_path))
    model = ActorCritic(num_states, num_actions)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/a3c_kungfu_master".format(opt.saved_path)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/a3c_kungfu_master".format(opt.saved_path),
                                         map_location=lambda storage, loc: storage))
    
    model.eval()
    
    episode_rewards = []
    
    for episode in range(10):
        state = torch.from_numpy(env.reset())
        done = True

        total_rewards_ep = 0

        while True:
            if done:
                h_0 = torch.zeros((1, 1024), dtype=torch.float)
                c_0 = torch.zeros((1, 1024), dtype=torch.float)
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()
            if torch.cuda.is_available():
                h_0 = h_0.cuda()
                c_0 = c_0.cuda()
                state = state.cuda()

            logits, value, h_0, c_0 = model(state, h_0, c_0)
            logits = logits / 255.0   # modification otherwise, always same action
            policy = F.softmax(logits, dim=1)
            m = Categorical(policy)
            action = m.sample().item()
            
#             action = torch.argmax(policy).item()
#             print(action)

            state, reward, done = env.step(action)
            total_rewards_ep += reward
            state = torch.from_numpy(state)
            if done:
                state = torch.from_numpy(env.reset())
                print("Game over")
                break

        print(total_rewards_ep)
        episode_rewards.append(total_rewards_ep)
        
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward
    
if __name__ == '__main__':
    opt = get_eval_args()
    mean_reward, std_reward = evaluate(opt)
    print(mean_reward, std_reward)
