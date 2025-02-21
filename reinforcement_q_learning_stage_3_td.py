# -*- coding: utf-8 -*-
"""
Reinforcement Learning (DQN) Tutorial
=====================================
**Author**: `Adam Paszke <https://github.com/apaszke>`_
            `Mark Towers <https://github.com/pseudo-rnd-thoughts>`_


This tutorial shows how to use PyTorch to train a Deep Q Learning (DQN) agent
on the CartPole-v1 task from `Gymnasium <https://gymnasium.farama.org>`__.

You might find it helpful to read the original `Deep Q Learning (DQN) <https://arxiv.org/abs/1312.5602>`__ paper

**Task**

The agent has to decide between two actions - moving the cart left or
right - so that the pole attached to it stays upright. You can find more
information about the environment and other more challenging environments at
`Gymnasium's website <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`__.

.. figure:: /_static/img/cartpole.gif
   :alt: CartPole

   CartPole

As the agent observes the current state of the environment and chooses
an action, the environment *transitions* to a new state, and also
returns a reward that indicates the consequences of the action. In this
task, rewards are +1 for every incremental timestep and the environment
terminates if the pole falls over too far or the cart moves more than 2.4
units away from center. This means better performing scenarios will run
for longer duration, accumulating larger return.

The CartPole task is designed so that the inputs to the agent are 4 real
values representing the environment state (position, velocity, etc.).
We take these 4 inputs without any scaling and pass them through a 
small fully-connected network with 2 outputs, one for each action. 
The network is trained to predict the expected value for each action, 
given the input state. The action with the highest expected value is 
then chosen.


**Packages**


First, let's import needed packages. Firstly, we need
`gymnasium <https://gymnasium.farama.org/>`__ for the environment,
installed by using `pip`. This is a fork of the original OpenAI
Gym project and maintained by the same team since Gym v0.19.
If you are running this in Google Colab, run:

.. code-block:: bash

   %%bash
   pip3 install gymnasium[classic_control]

We'll also use the following from PyTorch:

-  neural networks (``torch.nn``)
-  optimization (``torch.optim``)
-  automatic differentiation (``torch.autograd``)

"""

#import gym
import game2048
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

import tqdm
import numpy as np
import multiprocessing as mp

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
import pickle

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


######################################################################
# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classes:
#
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment. It essentially maps (state, action) pairs
#    to their (next_state, reward) result, with the state being the
#    screen difference image as described later on.
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class SumTree:
    """SumTree数据结构实现优先级的树状存储"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # 树节点数=2N-1
        self.data = np.zeros(capacity, dtype=object)  # 存储经验数据
        self.data_pointer = 0  # 数据指针
    
    def add(self, priority, data):
        """添加数据和优先级到树中"""
        tree_idx = self.data_pointer + self.capacity - 1  # 计算叶子节点位置
        self.data[self.data_pointer] = data  # 存储数据
        self.update(tree_idx, priority)  # 更新树中的优先级
        self.data_pointer = (self.data_pointer + 1) % self.capacity  # 循环覆盖
    
    def update(self, tree_idx, priority):
        """更新指定位置的优先级并传播变化"""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:  # 向上传播到根节点
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def get_leaf(self, v):
        """根据随机值v采样叶子节点"""
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
class PrioritizedReplayMemory(object):
    """带优先级的经验回放池"""
    def __init__(self, capacity=1000000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.alpha = alpha  # 优先级系数（0=均匀采样，1=完全优先级）
        self.beta = beta  # 重要性采样调整系数
        self.beta_increment = beta_increment
        self.abs_err_upper = 1.0  # 初始TD误差上限
        
        self.tree = SumTree(capacity)
    
    def push(self, *args, priority=None):
        """存储经验并初始化优先级（若未提供）"""
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])  # 现有最大优先级
        if max_priority == 0:  # 空树时初始化优先级
            max_priority = self.abs_err_upper
        if priority is None:  # 新经验的初始优先级设为当前最大
            priority = max_priority
        data = Transition(*args)
        self.tree.add(priority**self.alpha, data)  # 用alpha平滑优先级
    
    def sample(self, batch_size):
        """根据优先级采样并计算重要性采样权重"""
        batch_idx = []
        batch_weights = []
        batch_data = []
        
        segment = self.tree.tree[0] / batch_size  # 将优先级总和分成batch_size段
        
        self.beta = np.min([1., self.beta + self.beta_increment])  # 逐步增加beta
        
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.tree[0] + 1e-10  # 最小概率
        max_weight = (min_prob * self.tree.capacity) ** (-self.beta)  # 最大重要性采样权重
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(v)
            
            prob = priority / self.tree.tree[0]  # 当前样本的概率
            weight = (prob * self.tree.capacity) ** (-self.beta)  # 重要性采样权重
            weight /= max_weight  # 归一化
            
            batch_idx.append(idx)
            batch_weights.append(weight)
            batch_data.append(data)
        
        return batch_data, batch_idx, batch_weights
    
    def update_priorities(self, indices, errors):
        """用新的TD误差更新优先级"""
        errors = np.clip(np.abs(errors), 1e-4, self.abs_err_upper)  # 限制误差范围
        for idx, err in zip(indices, errors):
            self.tree.update(idx, err**self.alpha)
    
    def __len__(self):
        """当前存储的经验数量"""
        return min(self.tree.data_pointer, self.tree.capacity)


'''class ReplayMemory(object):

    def __init__(self,capacity=1000000):
        self.memory = []
        self.capacity=capacity

    def push(self, *args):
        """Save a transition"""
        if len(self.memory)>self.capacity:
            self.memory[random.randint(0,self.capacity-1)]=Transition(*args)
        else:
            self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)'''


######################################################################
# Now, let's define our model. But first, let's quickly recap what a DQN is.
#
# DQN algorithm
# -------------
#
# Our environment is deterministic, so all equations presented here are
# also formulated deterministically for the sake of simplicity. In the
# reinforcement learning literature, they would also contain expectations
# over stochastic transitions in the environment.
#
# Our aim will be to train a policy that tries to maximize the discounted,
# cumulative reward
# :math:`R_{t_0} = \sum_{t=t_0}^{\infty} \gamma^{t - t_0} r_t`, where
# :math:`R_{t_0}` is also known as the *return*. The discount,
# :math:`\gamma`, should be a constant between :math:`0` and :math:`1`
# that ensures the sum converges. A lower :math:`\gamma` makes 
# rewards from the uncertain far future less important for our agent 
# than the ones in the near future that it can be fairly confident 
# about. It also encourages agents to collect reward closer in time 
# than equivalent rewards that are temporally far away in the future.
#
# The main idea behind Q-learning is that if we had a function
# :math:`Q^*: State \times Action \rightarrow \mathbb{R}`, that could tell
# us what our return would be, if we were to take an action in a given
# state, then we could easily construct a policy that maximizes our
# rewards:
#
# .. math:: \pi^*(s) = \arg\!\max_a \ Q^*(s, a)
#
# However, we don't know everything about the world, so we don't have
# access to :math:`Q^*`. But, since neural networks are universal function
# approximators, we can simply create one and train it to resemble
# :math:`Q^*`.
#
# For our training update rule, we'll use a fact that every :math:`Q`
# function for some policy obeys the Bellman equation:
#
# .. math:: Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))
#
# The difference between the two sides of the equality is known as the
# temporal difference error, :math:`\delta`:
#
# .. math:: \delta = Q(s, a) - (r + \gamma \max_a' Q(s', a))
#
# To minimize this error, we will use the `Huber
# loss <https://en.wikipedia.org/wiki/Huber_loss>`__. The Huber loss acts
# like the mean squared error when the error is small, but like the mean
# absolute error when the error is large - this makes it more robust to
# outliers when the estimates of :math:`Q` are very noisy. We calculate
# this over a batch of transitions, :math:`B`, sampled from the replay
# memory:
#
# .. math::
#
#    \mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta)
#
# .. math::
#
#    \text{where} \quad \mathcal{L}(\delta) = \begin{cases}
#      \frac{1}{2}{\delta^2}  & \text{for } |\delta| \le 1, \\
#      |\delta| - \frac{1}{2} & \text{otherwise.}
#    \end{cases}
#
# Q-network
# ^^^^^^^^^
#
# Our model will be a feed forward  neural network that takes in the
# difference between the current and previous screen patches. It has two
# outputs, representing :math:`Q(s, \mathrm{left})` and
# :math:`Q(s, \mathrm{right})` (where :math:`s` is the input to the
# network). In effect, the network is trying to predict the *expected return* of
# taking each action given the current input.
#

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1)
        
        # 全连接层
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)  # 输出4个动作的Q值

    def forward(self, x):
        # 输入x的形状为 [batch_size, 1, 4, 4]
        x = F.relu(self.conv1(x))  # [batch_size, 16, 5, 5]
        x = F.relu(self.conv2(x))  # [batch_size, 32, 6, 6]
        x = F.relu(self.conv3(x))  # [batch_size, 64, 7, 7]
        x = x.view(x.size(0), -1)  # [batch_size, 64 * 7 * 7]
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 输出4个Q值
        return x

"""class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)"""


######################################################################
# Training
# --------
#
# Hyperparameters and utilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This cell instantiates our model and its optimizer, and defines some
# utilities:
#
# -  ``select_action`` - will select an action according to an epsilon
#    greedy policy. Simply put, we'll sometimes use our model for choosing
#    the action, and sometimes we'll just sample one uniformly. The
#    probability of choosing a random action will start at ``EPS_START``
#    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
#    controls the rate of the decay.
# -  ``plot_durations`` - a helper for plotting the duration of episodes,
#    along with an average over the last 100 episodes (the measure used in
#    the official evaluations). The plot will be underneath the cell
#    containing the main training loop, and will update after every
#    episode.
#

if __name__=="__main__":
    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    mp.set_start_method('spawn')
    manager = mp.Manager()
    shared_data = manager.Namespace()
    shared_data.EPS_START = 0.
    shared_data.EPS_END = 0.
    shared_data.EPS_DECAY = 1000
    shared_data.MAX_BATCH_SIZE = 64
    MAX_BATCH_SIZE = 64
    GAMMA = 0.999
    EPS_START = 0.0
    EPS_END = 0.0
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 3e-5
    RESUME=True

    env=game2048.gym_env()

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state = env.reset()
    n_observations = len(state)

    shared_data.policy_net = DQN().cpu().to(torch.device("cpu"))
    policy_net = DQN().to(torch.device("cuda"))
    target_net = DQN().to(torch.device("cuda"))
    if RESUME:
        shared_data.policy_net = torch.load("last_policy_net.pth", weights_only=False).cpu()
        policy_net = torch.load("last_policy_net.pth", weights_only=False)
        target_net = torch.load('last_target_net.pth',weights_only=False)
        """with open('replay_memory.pkl', 'rb') as f:
            memory = pickle.load(f)
        with open('steps_done.pkl', 'rb') as f:
            steps_done = pickle.load(f)"""
    else:
        target_net.load_state_dict(policy_net.state_dict())
    steps_done = 0

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)


def select_action(state,EPS_START,EPS_END,EPS_DECAY,steps_done,policy_net):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # 确保 state 的形状是 [1, 1, 4, 4]
            if len(state.shape) == 2:  # 如果 state 是 [4, 4]
                state = state.unsqueeze(0).unsqueeze(0)  # 调整为 [1, 1, 4, 4]
            elif len(state.shape) == 3:  # 如果 state 是 [1, 4, 4]
                state = state.unsqueeze(0)  # 调整为 [1, 1, 4, 4]
             # 获取 Q 值
            q_values = policy_net(state)
            
            # 选择动作
            action = q_values.max(1).indices
            action = action.view(-1, 1)  # 动态调整形状
            return action,steps_done
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long),steps_done


episode_scores = []


def plot_scores(show_result=False):
    plt.figure(1)
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score(2^k)')
    plt.plot(scores_t.numpy())
    # Take 64 episode averages and plot them too
    if len(scores_t) >= 64:
        means = scores_t.unfold(0, 64, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(63)+means[0], means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
    
    if len(scores_t) >= 64:
        return means[-1]
    else:
        return 0

    


######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By definition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state. We also use a target network to compute :math:`V(s_{t+1})` for
# added stability. The target network is updated at every step with a 
# `soft update <https://arxiv.org/pdf/1509.02971.pdf>`__ controlled by 
# the hyperparameter ``TAU``, which was previously defined.
#

def optimize_model(policy_net, target_net, optimizer, states, actions, rewards, next_states, GAMMA, device):
    # 确保输入张量和模型在同一设备上
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    state_action_values = policy_net(states.unsqueeze(1)).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_state_values = torch.zeros_like(rewards, device=device)
        non_final_mask = torch.tensor(tuple(s is not None for s in next_states), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s.unsqueeze(0) for s in next_states if s is not None]).to(device)
        next_state_values[non_final_mask] = target_net(non_final_next_states.unsqueeze(1)).max(1).values

    expected_state_action_values = rewards + GAMMA * next_state_values
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    td_errors = torch.abs(state_action_values - expected_state_action_values).detach().cpu().numpy()
    return td_errors

def worker(env, queue, event, device, done_flag,shared_data):
    local_memory = PrioritizedReplayMemory(capacity=100000)
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    terminated = False
    episode_score = 0  # 初始化得分
    steps_done=0
    start_time = time.time()
    while not done_flag.value:
        action,steps_done = select_action(state,shared_data.EPS_START,shared_data.EPS_END,shared_data.EPS_DECAY,steps_done,shared_data.policy_net)
        observation, reward, terminated = env.step(action.item(), with_wrong_move=True)
        real_action = torch.tensor([[reward[1]]], device=device, dtype=torch.long)
        reward = torch.tensor([reward[0]], device=device)

        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # 存储到本地内存
        if action.item() != real_action.item():
            local_memory.push(state, action, next_state, torch.tensor([-1], device=device), priority=1)
        local_memory.push(state, real_action, next_state, reward, priority=0.5)

        # 移动到下一个状态
        state = next_state

        # 将采样数据放入队列
        if len(local_memory) >= shared_data.MAX_BATCH_SIZE:
            transitions, indices, weights = local_memory.sample(shared_data.MAX_BATCH_SIZE)
            queue.put(("data", transitions, indices, weights))

            # 等待主进程的信号继续执行
            event.wait()
            event.clear()

        if terminated:
            print(f"线程完成一局游戏，耗时: {time.time() - start_time:.2f}秒")
            start_time = time.time()
            episode_score = np.log2(env.score)  # 记录最终得分
            queue.put(("score", episode_score))  # 将得分传递给主进程
            state = env.reset()  # 重置环境开始下一局
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            terminated = False

if __name__ == "__main__":
    # 多线程参数
    num_threads = 2  # 线程数量
    max_episodes = 4  # 每个线程需要完成的游戏局数
    queue = mp.Queue()  # 用于存储采样数据
    event = mp.Event()  # 用于同步 worker 线程
    done_flag = mp.Value('i', 0)  # 标记第一个线程是否完成指定局数
    best_score = 0  # 初始化最佳得分

    processes = []
    for _ in range(num_threads):
        p = mp.Process(target=worker, args=(env, queue, event, torch.device("cpu"), done_flag,shared_data))
        p.start()
        processes.append(p)

    episode_counter = 0  # 记录第一个线程完成的游戏局数
    max_score=0
    while True:
        # 从队列中获取所有线程的采样数据
        transitions, indices, weights = [],[],[]
        while not queue.empty():
            item = queue.get()
            if item[0] == "score":  # 如果是得分信息
                _, episode_score = item
                print(f"线程完成一局游戏，得分: {episode_score}")
                episode_scores.append(episode_score)
                mean_score=plot_scores()
                if episode_counter < max_episodes:
                    episode_counter += 1
                if mean_score>max_score:
                    torch.save(policy_net, 'best_policy_net.pth')
                    torch.save(target_net, 'best_target_net.pth')
                    max_score=mean_score
                    print(f"保存了新的最佳模型，得分: {best_score}")
            elif item[0] == "data":  # 如果是采样数据
                _, item_transitions, item_indices, item_weights = item
                transitions.extend(item_transitions)
                indices.extend(item_indices)
                weights.extend(item_weights)

        # 如果第一个线程完成指定局数，触发结束信号
        if episode_counter >= max_episodes:
            done_flag.value = 1
            break

        # 如果队列为空，继续等待
        if not transitions:
            time.sleep(0.1)
            continue

        # 合并所有线程的采样数据
        batch = Transition(*zip(*transitions))
        states = torch.cat(batch.state).to(device)
        actions = torch.cat(batch.action).to(device)
        rewards = torch.cat(batch.reward).to(device)
        next_states = torch.cat(batch.next_state).to(device)

        # 优化模型
        td_errors = optimize_model(policy_net, target_net, optimizer, states, actions, rewards, next_states, GAMMA, device)

        shared_data.policy_net.load_state_dict(policy_net.state_dict())
        shared_data.policy_net.cpu()

        # 通知所有线程继续执行下一步
        event.set()

    # 终止所有线程
    for p in processes:
        p.terminate()
        p.join()

    print("训练完成！")
    plot_scores(show_result=True)
    plt.ioff()
    plt.show()