import gym

env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')
print(env.action_space)
print(env.observation_space)

observation = env.reset()

for t in range(1000):
	env.render()
	print(observation)
	action = env.action_space.sample()  # take a random action
	observation, reward, done, info = env.step(action)
	if done:
		print("Episode finished after {} timesteps".format(t + 1))
		break
env.close()
