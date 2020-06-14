import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb

class Critic(nn.Module):

	def __init__(self, state_dim, action_dim):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of input action (int)
		:return:
		"""
		super(Critic, self).__init__()

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.conv1 = nn.Conv2d(1, 32, kernel_size = 5, stride = 2, padding = 2) #100 - 5 + 4/ 2 = 50
		self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 2) #50 -3/2 = 26
		self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 2) #24 -3/2 = 14
		self.linear1 = nn.Linear(64 * 14 * 14, 512)
		self.linear2 = nn.Linear(512, 128)

		self.fc1 = nn.Linear(action_dim, 256)
		self.fc2 = nn.Linear(256,128)

		self.fc3 = nn.Linear(256,1)

	def forward(self, x, a):
		"""
		returns Value function Q(s,a) obtained from critic network
		:param state: Input state (Torch Variable : [n,state_dim] )
		:param action: Input Action (Torch Variable : [n,action_dim] )
		:return: Value function : Q(S,a) (Torch Variable : [n,1] )
		"""

		x = F.relu(self.conv1(x.to(self.device)))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = x.view(-1, 64 * 14 * 14)
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))

		a = F.relu(self.fc1(a))
		a = F.relu(self.fc2(a))

		xa = torch.cat((x, a),dim=1)
		xa = self.fc3(xa)

		return xa


class Actor(nn.Module):

	def __init__(self, state_dim, action_dim):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of output action (int)
		:param action_lim: Used to limit action in [-action_lim,action_lim]
		:return:
		"""
		super(Actor, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self.conv1 = nn.Conv2d(1, 32, kernel_size = 5, stride = 2, padding = 2) #100 - 5 + 4/ 2 = 50
		self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 2) #50 -3/2 = 26
		self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 2) #24 -3/2 = 14
		self.linear1 = nn.Linear(64 * 14 * 14, 512)
		self.linear2 = nn.Linear(512, action_dim)

	def forward(self, x):

		"""
		returns policy function Pi(s) obtained from actor network
		this function is a gaussian prob distribution for all actions
		with mean lying in (-1,1) and sigma lying in (0,1)
		The sampled action can , then later be rescaled
		:param state: Input state (Torch Variable : [n,state_dim] )
		:return: Output action (Torch Variable: [n,action_dim] )
		"""

		try:
			x = F.relu(self.conv1(x.to(self.device)))
		except:
			print(x.shape)
			ipdb.set_trace()
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = x.view(-1, 64 * 14 * 14)
		x = F.relu(self.linear1(x))
		x = self.linear2(x)

		return x
