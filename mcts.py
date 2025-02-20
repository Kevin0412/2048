import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from game2048 import board,gym_env  # 导入game2048中的board和game类
import copy
from reinforcement_q_learning import DQN
import tqdm
import pickle
from datetime import datetime

class MCTSNode:
    def __init__(self, game_state:board, parent=None, dqn_model=None):
        self.game_state = game_state  # game2048中的board对象
        self.parent = parent
        self.children = defaultdict(dict)  # {action: {outcome: node}}
        self.visit_count = 0
        self.total_value = 0.0
        self.dqn_model = dqn_model  # 加载好的DQN模型
        
        # 预计算合法动作
        self.legal_actions = self._get_legal_actions()
        
        # 缓存DQN的Q值预测
        with torch.no_grad():
            state_tensor = torch.Tensor(self._state_to_input()).unsqueeze(0).unsqueeze(0)
            self.q_values = dqn_model(state_tensor).numpy().flatten()

    def _state_to_input(self):
        """将状态转换为DQN输入格式（归一化）"""
        processed = self.game_state.normalize_2d()
        return processed

    def _get_legal_actions(self):
        """返回可执行的动作列表（0:上, 1:左, 2:下, 3:右）"""
        actions = ["w", "a", "s", "d"]
        legal_actions = []
        for action in actions:
            if self.game_state.moveable(action):
                legal_actions.append(action)
        return legal_actions

    def select_action(self, c=1.5):
        """结合DQN先验的UCT选择"""
        best_score = -float('inf')
        best_action = None

        for action in self.legal_actions:
            # 获取该动作所有可能的结果节点
            outcome_nodes = self.children[action].values()
            total_visits = sum(n.visit_count for n in outcome_nodes)
            
            # 计算混合UCT值
            q_dqn = self.q_values[["w", "a", "s", "d"].index(action)]  # DQN的Q值
            exploitation = sum(n.total_value for n in outcome_nodes) / (total_visits + 1e-6)
            exploration = c * np.sqrt(np.log(self.visit_count + 1) / (total_visits + 1e-6))
            
            # 加权综合（DQN权重随访问次数衰减）
            alpha = 0.7 * (1 - self.visit_count/1000)  # 初始0.7，逐步降低
            score = alpha * q_dqn + (1-alpha) * (exploitation + exploration)
            
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action

    def expand(self):
        """扩展节点时考虑所有可能的随机结果"""
        selected_action = self.select_action()
        new_game_state = self.game_state.move(selected_action)  # 执行动作并获取新状态
        moved = new_game_state != self.game_state  # 检查是否移动
        
        if not moved:  # 无效动作
            return None
                
        # 生成所有可能的随机结果（考虑2和4的生成）
        empty_cells = [(i, j) for i in range(4) for j in range(4) if new_game_state[i][j] == 0]
        for (i, j) in empty_cells:
            for value in [1, 2]:  # 在game2048中，2和4分别用1和2表示
                child_state = copy.deepcopy(new_game_state)
                child_state[i][j] = value
                self.children[selected_action][(i, j, value)] = MCTSNode(child_state, self, self.dqn_model)
        return selected_action

    def simulate(self, rollout_depth=8):
        """使用DQN引导的混合模拟"""
        current_state = copy.deepcopy(self.game_state)
        total_reward = 0
        
        for _ in range(rollout_depth):
            if current_state.end():  # 判断游戏是否结束
                break
                
            # 80%概率使用DQN策略，20%随机
            if np.random.rand() < 0.0:  
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(current_state.normalize_2d()).unsqueeze(0).unsqueeze(0)
                    action_index = torch.argmax(self.dqn_model(state_tensor)).item()
                    action = ["w", "a", "s", "d"][action_index]
            else:
                action = np.random.choice(self.legal_actions)
                
            new_state = current_state.move(action)  # 执行动作
            moved = new_state != current_state  # 检查是否移动
            if not moved:
                continue
            
            # 随机生成新方块
            empty_cells = [(i, j) for i in range(4) for j in range(4) if new_state[i][j] == 0]
            if empty_cells:
                i, j = empty_cells[np.random.choice(len(empty_cells))]
                new_state[i][j] = 1 if np.random.rand() < 0.9 else 2  # 90%生成2，10%生成4
            
            reward = self._calculate_reward(current_state, new_state, action)
            total_reward += reward
            current_state = new_state
            
        return total_reward

    def backpropagate(self, value):
        """回溯更新时考虑随机分支的概率权重"""
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            if node.parent is None:
                break
                
            # 找到父节点到当前节点的路径及其概率
            for action in node.parent.children:
                for outcome, (prob, child) in node.parent.children[action].items():
                    if child is node:
                        value *= prob  # 根据路径概率衰减价值
                        break
            node = node.parent

    def _calculate_reward(self, old_state, new_state,action):
        """计算即时奖励"""
        """old_score = old_state.reward()
        new_score = new_state.reward()
        return new_score-old_score"""
        temp_env = gym_env()
        temp_env.board = old_state
        temp_env.step(action)
        return temp_env.score

class MCTS:
    def __init__(self, dqn_model, iterations=512):
        self.dqn_model = dqn_model
        self.iterations = iterations
        
    def search(self, root_state):
        root = MCTSNode(root_state, dqn_model=self.dqn_model)
        
        for _ in range(self.iterations):
            node = root
            path = []
            
            # 选择阶段
            while node.legal_actions:
                action = node.select_action()
                if action is None:
                    break
                
                # 获取该动作所有可能的结果节点
                outcome_nodes = node.children[action]
                if not outcome_nodes:
                    break  # 如果没有结果节点，则跳出循环
                
                total_visits = sum(child.visit_count for child in outcome_nodes.values())
                
                # 按概率选择随机结果分支
                outcomes = list(outcome_nodes.items())
                if not outcomes:
                    break  # 如果没有结果节点，则跳出循环
                
                probs = [prob for (out, (prob, child)) in outcomes]
                chosen_idx = np.random.choice(len(outcomes), p=probs)
                outcome, (prob, node) = outcomes[chosen_idx]
                path.append((outcome, node))
                
            # 扩展阶段
            if not node.children:
                action = node.expand()
                if action is not None:
                    outcome = next(iter(node.children[action]))
                    node = node.children[action][outcome][1]
                    path.append((outcome, node))
                    
            # 模拟阶段
            value = node.simulate()
            
            # 回溯阶段
            for outcome, child_node in reversed(path):
                child_node.backpropagate(value)
                value = child_node.total_value / child_node.visit_count
            
        # 选择最佳动作（考虑访问次数和Q值）
        best_action = None
        best_score = -float('inf')
        for action, children in root.children.items():
            total_visits = sum(child.visit_count for child in children.values())
            q_value = root.q_values[["w", "a", "s", "d"].index(action)]
            score = 0.7 * q_value + 0.3 * total_visits
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action

if __name__ == "__main__":
    # 加载预训练的DQN模型
    dqn_model = torch.load("models/7/best_policy_net.pth",weights_only=False, map_location=torch.device("cpu"))  # 确保模型路径正确
    dqn_model.eval()  # 设置为评估模式

    # 初始化MCTS
    mcts = MCTS(dqn_model=dqn_model, iterations=512)  # 每次搜索迭代500次

    # 运行多次游戏以评估性能
    num_games = 32  # 运行100次游戏
    scores = []
    max_tiles = []

    for i in tqdm.tqdm(range(num_games), desc="Playing games"):
        # 初始化游戏
        env = gym_env()
        env.reset()
        print(env.board)

        game_data = []

        while not env.end():
            # 使用MCTS获取最佳动作
            best_action = mcts.search(root_state=env.board)
            print(f"Best Action: {['上', '左', '下', '右'][['w', 'a', 's', 'd'].index(best_action)]}")
            
            # 保存当前状态和动作
            game_data.append({"state": copy.deepcopy(env.board), "action": best_action})
            
            # 执行动作
            env.step(best_action)
            print(env.board)

        # 保存游戏状态和动作到文件
        now_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mtcs_games/{env.score}_{now_time}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(game_data, f)

        # 记录游戏结果
        scores.append(env.score)
        max_tiles.append(env.board.max_tile())

    # 统计结果
    avg_score = np.mean(scores)
    avg_max_tile = np.mean(max_tiles)
    max_score = np.max(scores)
    max_max_tile = np.max(max_tiles)

    print(f"Average Score: {avg_score:.2f}")
    print(f"Average Max Tile: {avg_max_tile}")
    print(f"Max Score: {max_score}")
    print(f"Max Max Tile: {max_max_tile}")

    # 可选：绘制分数分布图
    import matplotlib.pyplot as plt

    plt.hist(scores, bins=20, color="skyblue", edgecolor="black")
    plt.title("Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.show()
    input()
