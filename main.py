#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# from __future__ import division
import numpy as np
from environment import environment
from agent import agent
import ipdb
from collections import deque
import random
#
import torch
from torch.autograd import Variable
import os
# import psutil
# import gc
import train
import buffer
#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#
MAX_EPISODES = 10000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
S_DIM = 100
A_DIM = 2
A_MAX = 10
ram = buffer.MemoryBuffer(MAX_BUFFER)
load_model = False
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram, load_model)
optimize_after_episode = 10
optimize_every = 5
#

start_point = 15, 90
end_point = 20, 90

obstacles = [
[(20,10) , (20, 50)],
[(20,50) , (90, 50)],
[(30,30) , (40, 40)],
]

env = environment(100, 100, obstacles, start_point, end_point)
reward_hist = deque(maxlen = 10)
episode_step_hist = deque(maxlen = 10)

for _ep in range(MAX_EPISODES):
    state = env.reset()
    done = False
    rewards = 0
    states = []
    actions = []
    new_states = []
    rwrds = []

    for step in range(100):

        if random.randint(0, 9) < 3: #explore 10% of the time
            action = trainer.get_exploitation_action(state)
        else:
            action = trainer.get_exploration_action(state)
        # action = [[1 , 0, 0]]

        new_state, reward, done = env.act(action)
        rewards += reward

        # env.plot_scene()

        new_state = np.float32(new_state)
        states.append(state)
        rwrds.append(reward)
        actions.append(action)
        new_states.append(new_state)

        state = new_state

        if done:
            break

    if rewards > -100: #only save positive memories
        for id in range(len(states)):
            ram.add(states[id], actions[id], rwrds[id], new_states[id])

    reward_hist.append(rewards)
    episode_step_hist.append(step)

    if (_ep >= optimize_after_episode) and ((_ep)%optimize_every ==0):

        if len(ram.buffer) > 128:
            loss_critic, loss_actor = trainer.optimize()

            print("episode " +str(_ep).ljust(5) +
            " | " + "Average Rewards " + str(round(sum(reward_hist)/len(reward_hist),1)).ljust(5) +
            " | " + "Average steps per episode " + str(round(sum(episode_step_hist)/len(episode_step_hist), 1)).ljust(5) +
            " | " + "Critic loss " + str(round(loss_critic.detach().item(), 3)).ljust(6) +
            " | " + "Actor loss " + str(round(loss_actor.detach().item(), 3)).ljust(6)
            )
            torch.save(trainer.critic, r'models/critic.pt')
            torch.save(trainer.actor, r'models/actor.pt')

        else:
            print("episode " +str(_ep).ljust(5) +
            " | " + "Average Rewards " + str(round(sum(reward_hist)/len(reward_hist),1)).ljust(5) +
            " | " + "Average steps per episode " + str(round(sum(episode_step_hist)/len(episode_step_hist), 1)).ljust(5)
            )


print("Fininshed")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
