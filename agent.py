"""
agent.py —— Q-learning 智能体模块（Agent）

这个模块负责：
1. 用一张 Q 表记住“每个状态下每个动作有多好”；
2. 根据 epsilon-greedy 策略选择动作（既探索又利用）；
3. 用 Q 更新公式把训练中获得的经验写回 Q 表。

核心概念：
- Q-table：二维表格，行是状态，列是动作。
  数值 Q(s,a) 越大，表示“在状态 s 做动作 a 越值得”；
- epsilon-greedy：以 epsilon 概率随机探索新动作，
  以 1-epsilon 概率选择当前 Q 值最大的动作（贪心利用）；
- Q 更新公式：
    Q(s,a) <- Q(s,a) + alpha * (r + gamma * max Q(s',a') - Q(s,a))
  其中 r + gamma * max Q(s',a') 叫“目标值 TD target”，
  它比旧的 Q(s,a) 更接近真实回报，所以把它当作更新方向。
"""

import random

import numpy as np


class QLearningAgent:
    """用 Q 表学习的智能体。"""

    def __init__(self, n_rows, n_cols, n_actions=4,
                 alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999,
                 seed=None):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_states = n_rows * n_cols
        self.n_actions = n_actions

        # 超参数（每个参数的含义在 README 里有完整解释）
        self.alpha = alpha            # 学习率：新信息占多大比重
        self.gamma = gamma            # 折扣因子：未来奖励打多少折
        self.epsilon = epsilon        # 当前探索概率
        self.epsilon_min = epsilon_min  # 探索概率下限
        self.epsilon_decay = epsilon_decay  # 每局结束后 epsilon 乘的衰减系数

        # 独立随机源：固定 seed 时结果可复现
        self.random = random.Random(seed)

        # Q 表：形状 = (状态数, 动作数)，初始全 0，表示“还没学过”
        self.q_table = np.zeros((self.n_states, self.n_actions))

    def state_to_index(self, state):
        """把 (row, col) 转成一维下标，方便查 Q 表。"""
        row, col = state
        return row * self.n_cols + col

    def choose_action(self, state, greedy=False):
        """选择动作。

        greedy=True：只利用不探索（评估和演示时用）；
        greedy=False：按 epsilon-greedy，前期多探索、后期多利用。
        """
        index = self.state_to_index(state)

        # 需要探索：随机选一个动作
        if not greedy and self.random.random() < self.epsilon:
            return self.random.randint(0, self.n_actions - 1)

        # 利用：选择 Q 值最大的动作（np.argmax 返回最大值的下标）
        return int(np.argmax(self.q_table[index]))

    def update(self, state, action, reward, next_state, done):
        """用一次 (state, action, reward, next_state, done) 经验更新 Q 表。"""
        s_idx = self.state_to_index(state)
        ns_idx = self.state_to_index(next_state)

        if done:
            # 本局结束，没有“下一个状态”，未来回报按 0 算
            best_next_q = 0.0
        else:
            best_next_q = float(np.max(self.q_table[ns_idx]))

        # 目标值：这次实际拿到的奖励 + 未来最优回报的折扣估计
        td_target = reward + self.gamma * best_next_q

        # 旧估计
        old_q = self.q_table[s_idx, action]

        # 更新：让旧 Q 值朝目标值方向移动一小步
        self.q_table[s_idx, action] = old_q + self.alpha * (td_target - old_q)

    def decay_epsilon(self):
        """每局结束后让探索概率变小，后期更多依赖学到的策略。"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
