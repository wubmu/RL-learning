import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from rl_hand.memory_common import ReplayMemory
from rl_hand.model_common import Qnet

class DQN(object):
    def __init__(self,env, state_dim, hidden_dim, action_dim,
                 memory_capacity=10000, max_steps=10000,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 target_update=10, target_tau=0.01, learning_rate=2e-3,
                 epsilon=0.01,
                 gpu_id=0):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim


        # reward设定
        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.done_penalty = done_penalty

        self.replay_buffer = ReplayMemory(memory_capacity)
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(self.device)



        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)  # 使用Adam优化器

        self.epsilon = epsilon  # epsilon-greedy
        self.target_update = target_update  # 目标网络更新频率
        self.target_tau = target_tau
        self.count = 0  # 计数器，记录更新次数

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()  # 动作索引的
        return action

    def learn(self, batch_size):
        # transition_dict = self.replay_buffer.sample(batch_size)
        b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(batch_size)
        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}

        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        # [state_size,action_dim] -> [state_size,1]
        max_next_q_values = self.target_q_net(next_states).detach().max(1)[0].view(-1, 1)

        q_targets = rewards + self.reward_gamma * max_next_q_values * (1- dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))

        self.optimizer.zero_grad()
        dqn_loss.backward() # 反向传播更新参数
        self.optimizer.step()

        # 更新target 网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict()) # 更新目标网络
        self.count += 1

    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)

    def evaluation(self, env, eval_episodes=10):
        rewards = []
        infos = []
        for i in range(eval_episodes):
            rewards_i = []
            infos_i = []
            state = env.reset()
            action = self.take_action(state)
            state, reward, done, info = env.step(action)
            done = done[0] if isinstance(done, list) else done
            rewards_i.append(reward)
            infos_i.append(info)
            while not done:
                action = self.take_action(state)
                state, reward, done, info = env.step(action)
                done = done[0] if isinstance(done, list) else done
                rewards_i.append(reward)
                infos_i.append(info)
            rewards.append(rewards_i)
            infos.append(infos_i)
        return rewards, infos


