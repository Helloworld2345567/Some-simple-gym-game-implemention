import argparse
from collections import namedtuple
from itertools import count
import pickle
import time
import matplotlib.pyplot as plt 
from matplotlib import animation 
import os, random
import numpy as np
from collections import deque

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from sac_agent import SAC
from replay_memory import ReplayMemory

from datetime import datetime



'''
Implementation of soft actor critic, dual Q network version 
Original paper: https://arxiv.org/abs/1801.01290
Not the author's implementation !
'''

gamma=0.99
batch_size=256
lr=3e-4
hidden_size=512
tau=0.005
alpha=0.2
start_steps=10000
update_start_steps=1e4
reward_scale = 10
eval_rate=0.8

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()

def display_frames_as_gif(frames):
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=1)
    anim.save('./gif/result0.gif', writer='ffmpeg', fps=30)
    
def init_parser():

    parser.add_argument("--env_name", default="BipedalWalker-v3")  # OpenAI gym environment name

    parser.add_argument('--capacity', default=5000000, type=int) # replay buffer size
    parser.add_argument('--iteration', default=100000, type=int) #  num of  games
    parser.add_argument('--batch_size', default=256, type=int) # mini batch size

    # optional parameters
    parser.add_argument('--render', default=False, type=bool) # show UI or not
    parser.add_argument('--train', default=False, type=bool) 
    parser.add_argument('--eval', default=True, type=bool) 
    parser.add_argument('--load', default=False, type=bool) # load model
    parser.add_argument('--render_interval', default=0, type=int) # after render_interval, the env.render() will work

init_parser()
args = parser.parse_args()

env = gym.make(args.env_name, render_mode="rgb_array", hardcore=False)

# Set seeds
#env.seed(args.seed)
#env.action_space.seed(args.seed)
#torch.manual_seed(args.seed)
#np.random.seed(args.seed)

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

def main():
    agent = SAC(state_dim, env.action_space, device, hidden_size, lr, gamma, tau, alpha)
    replay_buffer = ReplayMemory(args.capacity)

    if args.train: print("Train True")
    if args.load: 
        print("Load True")
        # agent.load_model(actor_path="./models_hard1/actor.pth", critic_path="./models_hard1/critic.pth")
        agent.load_model(actor_path='models/actor3_hard.pth', critic_path='models/critic3_hard.pth')
        #agent.load_model(actor_path='models/actor2.pth', critic_path='models/critic2.pth')
        
    max_reward = 3200
    updates = 0
    avg_reward = 0.
    total_steps = 0
    count_3000 = 0
    time_start = time.time()
    scores_deque = deque(maxlen=100)
    avg_scores_array = []
    frames=[]
    rewards_log = []
    average_log = []
    for i in range(1,args.iteration+1):
        ep_r = 0
        ep_s = 0
        
        done = False
        state = env.reset()[0]
        while not done:
            action = []            
            if total_steps < start_steps and not args.load:
                action = env.action_space.sample()
            else:
                use_eval = False
                if args.render:
                    use_eval = True 
                else:
                    if np.random.rand(1) > eval_rate:          #取用预测
                        use_eval = True
                action = agent.select_action(state, use_eval)
                
            next_state, reward, done, _, _ = env.step(action)
            if reward <=-100:
                reward=-1
            reward = reward * reward_scale
            
            ep_r += reward
            ep_s += 1
            total_steps += 1
            
            if args.render and i >= args.render_interval : 
                image=env.render()
                frames.append(image)
            
            if ep_s >= 1600:
                mask = 1
                if ep_s ==2500:
                    done = True
            else:
                if not done:
                    mask = 1.0
                else:
                    mask = 0.0
            if args.train:
                replay_buffer.push(state, action, reward, next_state, mask)
            state = next_state           
        if args.train:
            for upi in range(ep_s):
                if args.load:
                    if len(replay_buffer) >= 10000:
                        agent.update_parameters(replay_buffer, batch_size, updates)
                        updates += 1
                if not args.load and len(replay_buffer) >= update_start_steps:
                    agent.update_parameters(replay_buffer, batch_size, updates)
                    updates += 1
        
        s =  (int)(time.time() - time_start)
        if ep_r > max_reward:
            max_reward = ep_r
            
        rewards_log.append(ep_r)    
        average_log.append(np.mean(rewards_log[-100:]))
        print("\rEp.: {}, Total Steps: {}, Ep.Steps: {}, Score: {:.2f},average: {:.2f}, Time: {:02}:{:02}  ".\
            format(i, total_steps, ep_s, ep_r, average_log[-1], \
                  s//3600, s%3600//60),end='')
        
        if ep_r >= 3200:
            count_3000 += 1
            if count_3000 >= 500 and average_log[-1]>3100:
                agent.save_model(actor_path='models/actor3_hard.pth', critic_path='models/critic3_hard.pth')
                break
        if i % 200 == 0:
            agent.save_model(actor_path='models/actor3_hard.pth', critic_path='models/critic3_hard.pth')
            print('Max_reward=',max_reward)
    if args.train:        
        np.save('./logs/rewards_log.npy', rewards_log)
        np.save('./logs/average_log.npy', rewards_log)
    if args.render:        
        display_frames_as_gif(frames)
    env.close()
    
if __name__ == '__main__':
    main()
    