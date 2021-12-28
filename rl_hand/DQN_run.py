from rl_hand.DQN import DQN
import random
import gym
import numpy as np
import torch
import torch.nn.functional as F

lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
env_name = 'CartPole-v0'
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
# agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
agent = DQN(env=env, state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim,
            learning_rate=lr, reward_gamma=gamma, epsilon=epsilon, target_update=target_update)

return_list = []
for i_episode in range(num_episodes):
    episode_return = 0
    state = env.reset()
    done = False
    while not done:
        action = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.replay_buffer._push_one(state, action, reward, next_state, done)
        episode_return += reward  # 这里的会报的计算不计算折扣因子衰减
        state = next_state

        if agent.replay_buffer.__len__() > minimal_size:  # 当buffer数据量超过一定值后，才进行Q网络训练
            b_s, b_a, b_r, b_ns, b_d = agent.replay_buffer.sample(batch_size)
            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
            agent.learn(batch_size)
    return_list.append(episode_return)
    if (i_episode + 1) % 20 == 0:
        print("Episode: {}, Score: {}".format(i_episode + 1, np.mean(return_list[-10:])))
