import os
import math
import numpy as np
import pickle
from game2048 import gym_env, board
import matplotlib.pyplot as plt


def get_game_count(folder_path,show=False):

    file_names = os.listdir(folder_path)

    # 步骤2：从每个文件名中提取最前面的整数
    game_counts = len(file_names)
    print(f"总局数: {game_counts}")
    integers = []
    #max_tile_distribution = []
    for file_name in file_names:
        if file_name.endswith('.pkl'):
            integer_str = file_name.split('_')[0]
            integers.append(int(integer_str))
            # 打开.pkl文件并读取内容
            '''with open(folder_path+"/"+file_name, 'rb') as file:  # 注意使用'rb'模式
                data = pickle.load(file)
                max_tile_distribution.append(data[-1]['state'].max_tile())'''

    #max_tile_counts = {2**i: max_tile_distribution.count(2**i) for i in range(1, 18)}

    ''' print("Max Tile Distribution:")
    for tile, count in max_tile_counts.items():
        print(f"{tile}: {count}")'''

    # 步骤3：计算每个整数的对数（以2为底）
    log2_values = [math.log2(i) for i in integers]

    # 步骤4：求这些对数值的平均值和标准差
    mean_log2 = np.mean(log2_values)
    std_dev_log2 = np.std(log2_values, ddof=1)

    # 输出结果
    print(f"平均值 ± 标准差: {mean_log2} ± {std_dev_log2}")

    if show:
        # 统计得分log2(score.score)的分布
        plt.hist(log2_values, bins=np.arange(min(log2_values), max(log2_values) + 0.1, 0.1), edgecolor='black')
        plt.xlabel('log2(score)')
        plt.ylabel('Frequency')
        plt.title('Distribution of log2(score)')
        plt.show()
        input()
    return log2_values

if __name__ == '__main__':
    # 步骤1：读取文件夹中的所有文件名
    folder_path = 'mcts_games/4'  # 替换为你的文件夹路径
    get_game_count(folder_path, show=True)
