import numpy as np
from scipy.stats import ttest_ind
from eval_mcts_result import get_game_count

# 示例数据
list1 = get_game_count('mcts_games/4')
list2 = [np.log2(28040), np.log2(28204), np.log2(16888), np.log2(16172), np.log2(32256)]

# 进行 Welch's t 检验
t_stat, p_value = ttest_ind(list1, list2, equal_var=False)

# 计算自由度 df
n1, n2 = len(list1), len(list2)
var1, var2 = np.var(list1, ddof=1), np.var(list2, ddof=1)
df = (var1 / n1 + var2 / n2) ** 2 / ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1))

# 输出结果
print(f"t 统计量: {t_stat}")
print(f"自由度 df: {df}")
print(f"p 值: {p_value}")