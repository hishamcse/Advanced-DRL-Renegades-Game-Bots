import os
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import torch
from model import ActorCritic, IntrinsicCuriosityModule
from optimizer import GlobalAdam
from process import local_train
import torch.multiprocessing as _mp
import shutil
import gym

def get_train_args():
    parser = argparse.ArgumentParser("A3C_ICM_KungFU_Master")
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--sigma', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--lambda_', type=float, default=0.1, help='a3c loss coefficient')
    parser.add_argument('--eta', type=float, default=0.2, help='intrinsic coefficient')
    parser.add_argument('--beta', type=float, default=0.2, help='curiosity coefficient')
    parser.add_argument("--num_local_steps", type=int, default=50)
    parser.add_argument("--num_global_steps", type=int, default=1e7)   # modified for kungfu 
    parser.add_argument("--num_processes", type=int, default=3)
    parser.add_argument("--save_interval", type=int, default=100, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=500, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/a3c_icm_kungfu_master")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--load_prev", type=bool, default=True)
#     args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    return args

def train(opt):
    torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    mp = _mp.get_context("spawn")
    
    tmp = gym.make("ALE/KungFuMaster-v5", render_mode="rgb_array").action_space.n
    global_model = ActorCritic(num_inputs=1, num_actions=tmp)
    global_icm = IntrinsicCuriosityModule(num_inputs=1, num_actions=tmp)
    
    if opt.load_prev:
        if opt.use_gpu:
            global_model.load_state_dict(torch.load("{}/a3c_kungfu_master".format(opt.saved_path)))
            global_icm.load_state_dict(torch.load("{}/icm_kungfu_master".format(opt.saved_path)))
        else:
            global_model.load_state_dict(torch.load("{}/a3c_kungfu_master".format(opt.saved_path),
                                             map_location=lambda storage, loc: storage))
            global_icm.load_state_dict(torch.load("{}/icm_kungfu_master".format(opt.saved_path),
                                             map_location=lambda storage, loc: storage))
        
    if opt.use_gpu:
        global_model.cuda()
        global_icm.cuda()
        
    global_model.share_memory()
    global_icm.share_memory()
    
#     # modification
#     global_model.train()
#     global_icm.train()

    optimizer = GlobalAdam(list(global_model.parameters()) + list(global_icm.parameters()), lr=opt.lr)
    processes = []
    for index in range(opt.num_processes):
        if index == 0:
            process = mp.Process(target=local_train, args=(index, opt, global_model, global_icm, optimizer, True))
        else:
            process = mp.Process(target=local_train, args=(index, opt, global_model, global_icm, optimizer, True))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()

if __name__ == '__main__':
    opt = get_train_args()
    print(opt)
    train(opt)
