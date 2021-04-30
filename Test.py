import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from collections import namedtuple, deque
import random
import gym
import matplotlib.pyplot as plt
from ddpg_full import *
import cv2


batch_size = 5
n_epochs = 4
alpha = 0.0003
env = gym.make('HalfCheetah-v2')
# Validate the training result from the files that have been saved in main.py
#####################################
agent = Agent(alpha=0.0025, beta=0.025, input_dims=[17], tau=0.001, env=env,
              batch_size=256,  layer1_size=256, layer2_size=256, n_actions=6)


# Validate results
agent.actor.load_state_dict(torch.load('checkpoint_actor.pth'))
agent.critic.load_state_dict(torch.load('checkpoint_critic.pth'))


state = env.reset()
for t in range(500):
    act = agent.choose_action(state)
    new_state, reward, done, info = env.step(act)
    env.render()
    state, reward, done, _ = env.step(act)
    if done:
        break

env.close()

