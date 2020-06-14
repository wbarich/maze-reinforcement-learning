#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import ipdb
import numpy as np

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ActorCritic(nn.Module):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, state_dim, action_dim, action_std):

        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.c_conv1 = nn.Conv2d(1, 32, kernel_size = 5, stride = 2, padding = 2) #100 - 5 + 4/ 2 = 50
        self.c_conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 2) #50 -3/2 = 26
        self.c_conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 2) #24 -3/2 = 14
        self.c_linear1 = nn.Linear(64 * 14 * 14, 512)
        self.c_linear2 = nn.Linear(512, 128)
        self.c_fc1 = nn.Linear(128, 1)

        self.a_conv1 = nn.Conv2d(1, 32, kernel_size = 5, stride = 2, padding = 2) #100 - 5 + 4/ 2 = 50
        self.a_conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 2) #50 -3/2 = 26
        self.a_conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 2) #24 -3/2 = 14
        self.a_linear1 = nn.Linear(64 * 14 * 14, 512)
        self.a_linear2 = nn.Linear(512, action_dim)

        self.action_var = torch.full((action_dim,), action_std*action_std).to(self.device)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def actor_forward(self, x):

        x = F.relu(self.a_conv1(x.to(self.device)))
        x = F.relu(self.a_conv2(x))
        x = F.relu(self.a_conv3(x))
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.a_linear1(x))
        x = self.a_linear2(x)

        return x

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def critic_forward(self, x):

        x = F.relu(self.c_conv1(x.to(self.device)))
        x = F.relu(self.c_conv2(x))
        x = F.relu(self.c_conv3(x))
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.c_linear1(x))
        x = F.relu(self.c_linear2(x))
        x = F.relu(self.c_fc1(x))

        return x

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def act(self, state, memory):
        state = state.unsqueeze(0)
        state = state.unsqueeze(0)
        action_mean = self.actor_forward(state)
        cov_mat = torch.diag(self.action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor_forward(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic_forward(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
