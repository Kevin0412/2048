import matplotlib.pyplot as plt
import numpy as np
from eval_mcts_result import get_game_count

# 数据
list1 = get_game_count('mcts_games/2')
list2 = get_game_count('ai_games/2')
list3 = get_game_count('mcts_games/4')
list4 = [np.log2(28040), np.log2(28204), np.log2(16888), np.log2(16172), np.log2(32256)]

# 绘制箱线图
plt.boxplot([list1, list2, list3, list4], labels=['MCTS', 'DQN', 'MCTS+DQN', 'HUMAN'])
plt.title('Boxplot Comparison')
plt.ylabel('log2(score)')
plt.xlabel('Groups')
plt.grid(True)
plt.show()