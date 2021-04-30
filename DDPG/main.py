  
from ddpg_full import Agent
import gym
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
#from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(1)

def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('average return')       
    plt.xlabel('episode')                     
    plt.plot(x, running_avg)
    plt.savefig(filename)

#writer = SummaryWriter()
env = gym.make('HalfCheetah-v2')
agent = Agent(alpha=0.0025, beta=0.025, input_dims=[17], tau=0.001, env=env,
              batch_size=256,  layer1_size=256, layer2_size=256, n_actions=6)

#agent.load_models()
np.random.seed(0)

score_history = []
for i in range(2000):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
        #env.render()
    score_history.append(score)
    #writer.add_scalar('average score', score_history, i)

    if i % 25 == 0:
        agent.save_models()
        torch.save(agent.actor.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic.state_dict(), 'checkpoint_critic.pth')

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 episode avg %.3f' % np.mean(score_history[-100:]))

filename = 'HalfCheetah.png'
plotLearning(score_history, filename, window=100)