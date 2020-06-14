import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import ipdb
import numpy as np
from environment import environment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
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
        # self.c_fc1 = nn.Linear(action_dim, 256)
        self.c_fc2 = nn.Linear(256,128)
        self.c_fc3 = nn.Linear(256,1)

        self.a_conv1 = nn.Conv2d(1, 32, kernel_size = 5, stride = 2, padding = 2) #100 - 5 + 4/ 2 = 50
        self.a_conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 2) #50 -3/2 = 26
        self.a_conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 2) #24 -3/2 = 14
        self.a_linear1 = nn.Linear(64 * 14 * 14, 512)
        self.a_linear2 = nn.Linear(512, action_dim)

        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)

    def actor_forward(self, x):
        try:
            # x = x.unsqueeze(0)
            # x = x.unsqueeze(0)
            x = F.relu(self.a_conv1(x.to(self.device)))
        except:
            ipdb.set_trace()
        x = F.relu(self.a_conv2(x))
        x = F.relu(self.a_conv3(x))
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.a_linear1(x))
        x = self.a_linear2(x)
        return x

    def critic_forward(self, x):

        x = F.relu(self.c_conv1(x.to(self.device)))
        x = F.relu(self.c_conv2(x))
        x = F.relu(self.c_conv3(x))
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.c_linear1(x))
        x = F.relu(self.c_linear2(x))
        x = F.relu(self.c_fc1(x))

        # a = F.relu(self.c_fc1(a))
        # a = F.relu(self.c_fc2(a))

        # xa = torch.cat((x, a),dim=1)
        # xa = self.c_fc3(xa)

        return x


    def act(self, state, memory):
        state = state.unsqueeze(0)
        state = state.unsqueeze(0)
        # print("asdasd")
        action_mean = self.actor_forward(state)
        cov_mat = torch.diag(self.action_var).to(device)

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
        cov_mat = torch.diag_embed(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic_forward(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()

        # Optimize policy for K epochs:

        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

def main():
    ############## Hyperparameters ##############

    solved_reward = 300         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 100        # max timesteps in one episode

    update_timestep = 4000      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 10               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)

    random_seed = None
    #############################################

    obstacles = [
    [(20,10) , (20, 50)],
    [(20,50) , (90, 50)],
    [(30,30) , (40, 40)],
    ]
    start_point = 15, 90
    end_point = 20, 90

    # creating environment
    env = environment(100, 100, obstacles, start_point, end_point)
    state_dim = 100
    action_dim = 2

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done = env.step(action)

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward

            if done:
                break

        avg_length += t

        # save every 500 episodes
        if i_episode % 200 == 0:
            torch.save(ppo.policy.state_dict(), 'models/agent_model.pth')

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

if __name__ == '__main__':
    main()
