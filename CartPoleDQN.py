import gym
import math
import random
import numpy as np

from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ENV_NAME = 'MountainCar-v0'
# ENV_NAME = 'MountainCarContinuous-v0'
ENV_NAME = 'CartPole-v0'
# ENV_NAME = 'CartPole-v1'
# ENV_NAME = 'LunarLander-v2'

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

NUM_EPISODES = 1000

steps_done = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 0.03

Transition = namedtuple(
	'Transition',
	('state', 'action', 'next_state', 'reward')
)


class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


class DQN(nn.Module):

	def __init__(self, state_n, action_n):
		super(DQN, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(state_n, 64),
			nn.Sigmoid(),
			nn.Linear(64, 32),
			nn.Sigmoid(),
			nn.Linear(32, 32),
			nn.Sigmoid(),
			nn.Linear(32, 16),
			nn.Sigmoid(),
			nn.Linear(16, action_n),
			nn.Sigmoid()
		)

		# self.linear_1 = nn.Linear(state_n, 64)
		# self.linear_2 = nn.Linear(64, 32)
		# self.linear_3 = nn.Linear(32, 32)
		# self.linear_4 = nn.Linear(32, 16)
		# self.linear_5 = nn.Linear(16, action_n)

	# Called with either one element to determine next action, or a batch
	# during optimization. Returns tensor([[left0exp,right0exp]...]).
	def forward(self, x):
		# x = torch.sigmoid(self.linear_1(x))
		# x = torch.sigmoid(self.linear_2(x))
		# x = torch.sigmoid(self.linear_3(x))
		# x = torch.sigmoid(self.linear_4(x))
		# x = torch.sigmoid(self.linear_5(x))
		return self.fc(x)


def select_action(model, state, action_space=None):
	global steps_done
	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END) * \
					math.exp(-1. * steps_done / EPS_DECAY)
	steps_done += 1
	if sample > eps_threshold:
		with torch.no_grad():
			# t.max(1) will return largest column value of each row.
			# second column on max result is index of where max element was
			# found, so we pick action with the larger expected reward.
			action = model(state)
			action = torch.argmax(action, dim=1, keepdim=True)
			action = action.view(-1, 1)
			return action
	else:
		action = torch.tensor([[random.randrange(action_space.n)]], device=device, dtype=torch.long)
		return action


loss_training = []


def optimize_model(policy_net, target_net, memory, optimizer):
	if len(memory) < BATCH_SIZE:
		return
	transitions = memory.sample(BATCH_SIZE)
	# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
	# detailed explanation). This converts batch-array of Transitions
	# to Transition of batch-arrays.
	batch = Transition(*zip(*transitions))

	# Compute a mask of non-final states and concatenate the batch elements
	# (a final state would've been the one after which simulation ended)
	non_final_mask = torch.tensor(
		tuple(map(lambda s: s is not None, batch.next_state)),
		device=device, dtype=torch.bool)
	non_final_next_states = torch.cat(
		[s for s in batch.next_state if s is not None])
	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
	# columns of actions taken. These are the actions which would've been taken
	# for each batch state according to policy_net
	state_action_values = policy_net(state_batch).gather(1, action_batch)

	# Compute V(s_{t+1}) for all next states.
	# Expected values of actions for non_final_next_states are computed based
	# on the "older" target_net; selecting their best reward with max(1)[0].
	# This is merged based on the mask, such that we'll have either the expected
	# state value or 0 in case the state was final.
	next_state_values = torch.zeros(BATCH_SIZE, device=device)
	next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
	# Compute the expected Q values
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch

	# Compute Huber loss
	loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

	loss_item = loss.item()
	loss_training.append(loss_item)

	# Optimize the model
	optimizer.zero_grad()
	loss.backward()
	for param in policy_net.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()

	return loss_item


def main():
	env = gym.make(ENV_NAME)
	env._max_episode_steps = 10000

	n_states = env.observation_space.shape[0]
	action_space = env.action_space
	n_actions = action_space.n

	policy_net = DQN(n_states, n_actions).to(device)
	target_net = DQN(n_states, n_actions).to(device)
	target_net.load_state_dict(policy_net.state_dict())
	target_net.eval()

	# optimizer = optim.RMSprop(policy_net.parameters(), lr=LEARNING_RATE)
	optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.003)

	memory = ReplayMemory(10000)

	for i_episode in range(NUM_EPISODES):
		# Initialize the environment and state
		state = torch.Tensor(env.reset()).to(device).view(-1, n_states)
		total_reward = 0

		# last_screen = get_screen()
		# current_screen = get_screen()
		# state = current_screen - last_screen
		for t in count():
			env.render()
			# Select and perform an action
			action = select_action(policy_net, state, action_space)
			next_state, reward, done, _ = env.step(action.item())

			done = math.fabs(next_state[0]) > 2

			total_reward += reward
			reward = torch.tensor([reward], device=device)
			next_state = torch.Tensor(next_state).to(device).view(-1, n_states)

			# Observe new state
			# last_screen = current_screen
			# current_screen = get_screen()
			# if not done:
			# 	next_state = current_screen - last_screen
			# else:
			# 	next_state = None

			# Store the transition in memory
			memory.push(state, action, next_state, reward)

			# Perform one step of the optimization (on the target network)
			loss = optimize_model(policy_net, target_net, memory, optimizer)
			# print(f"reward: {reward.item()} loss: {loss}")
			if done:
				# episode_durations.append(t + 1)
				# plot_durations()
				print(f"E {i_episode} finished at {t+1}, reward: {total_reward}")
				break

			# Move to the next state
			state = next_state

		# Update the target network, copying all weights and biases in DQN
		if i_episode % TARGET_UPDATE == 0:
			target_net.load_state_dict(policy_net.state_dict())

	print('Complete')
	env.render()
	env.close()
	# plt.ioff()
	# plt.show()


if __name__ == '__main__':
	main()
