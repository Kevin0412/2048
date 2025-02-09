import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from game2048 import gym_env,board
from reinforcement_q_learning import DQN
import math
import statistics
import tqdm
import copy

class Airun:
    def __init__(self, model_path='best_policy_net.pth'):
        # 直接加载整个模型
        self.model = torch.load(model_path, weights_only=False).to("cpu")
        self.model.eval()  # 设置为评估模式
        self.env = gym_env()

    def preprocess_state(self, state):
        # 将状态转换为模型输入的格式
        state = np.array(state)
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
        return state

    def choose_action(self, state):
        with torch.no_grad():
            state = self.preprocess_state(state)
            q_values = self.model(state)
            action = torch.argmax(q_values).item()  # 选择Q值最大的动作
        return action

    def print_game_state(self, state, action, total_reward):
        # 打印当前游戏状态、动作和总奖励
        print("\n当前游戏状态：")
        print(self.env.board)  # 将状态转换为 4x4 的二维数组并打印
        print(f"选择动作: {['上', '左', '下', '右'][action]}")
        print(f"当前总奖励: {total_reward}")
        print("-----------------------------")

    def run(self,step=False,show=True):
        state = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            action = self.choose_action(state)
            next_state, reward, done = self.env.step(action)
            total_reward += reward

            # 打印当前游戏状态
            if show:
                self.print_game_state(state, action, total_reward)
                if step:
                    input()

            state = next_state
        if show:
            print(self.env)
        return self.env

if __name__ == "__main__":
    airun = Airun(model_path='best_policy_net.pth')
    scores=[]
    for i in tqdm.tqdm(range(2048)):
        scores.append(copy.deepcopy(airun.run(show=False)))
    scores.sort(key=lambda x: x.score, reverse=True)
    log_scores = [math.log2(score.score) for score in scores]
    print(scores[0])
    mean_log_score = statistics.mean(log_scores)
    stdev_log_score = statistics.stdev(log_scores)
    print(mean_log_score, "±", stdev_log_score)
    import matplotlib.pyplot as plt

    # 统计2**max(score.board.flatten())的分布
    max_tile_distribution = [score.board.max_tile() for score in scores]
    max_tile_counts = {2**i: max_tile_distribution.count(2**i) for i in range(1, 18)}

    print("Max Tile Distribution:")
    for tile, count in max_tile_counts.items():
        print(f"{tile}: {count}")

    # 统计得分log2(score.score)的分布
    plt.hist(log_scores, bins=np.arange(min(log_scores), max(log_scores) + 0.1, 0.1), edgecolor='black')
    plt.xlabel('log2(score)')
    plt.ylabel('Frequency')
    plt.title('Distribution of log2(score)')
    plt.show()
    input()

    
