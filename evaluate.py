"""
evaluate.py —— 评估程序

读取 train.py 保存的 Q 表，用“纯贪心策略”（epsilon=0，只选 Q 值最大的动作）
连续跑 100 局，统计成功率和平均步数。
同时跑 100 局随机策略作为对照，证明 Q-learning 学到的策略明显优于乱走。

运行：python evaluate.py（需要先运行 python train.py）
"""

import os

import numpy as np

from env import GridMazeEnv
from agent import QLearningAgent

N_EVAL = 100
MAX_STEPS = 150
RESULTS_DIR = "results"


def run_policy(agent, greedy):
    """按给定策略跑一局，返回 (是否成功, 走的步数)。"""
    env = GridMazeEnv()
    state = env.reset()
    for step in range(MAX_STEPS):
        action = agent.choose_action(state, greedy=greedy)
        state, _, done = env.step(action)
        if done:
            break
    success = state == env.goal_pos
    return success, step + 1


def main():
    q_path = os.path.join(RESULTS_DIR, "q_table.npy")
    if not os.path.exists(q_path):
        raise FileNotFoundError("没有找到 results/q_table.npy，请先运行 python train.py")

    q_table = np.load(q_path)

    env = GridMazeEnv()
    trained_agent = QLearningAgent(env.rows, env.cols, n_actions=len(env.ACTIONS))
    trained_agent.q_table = q_table   # 把训练好的 Q 表装回智能体

    # 随机策略对照：epsilon=1.0 表示每一步都随机选择动作
    random_agent = QLearningAgent(env.rows, env.cols,
                                  n_actions=len(env.ACTIONS), epsilon=1.0)

    trained_success = []
    trained_steps = []
    for _ in range(N_EVAL):
        success, steps = run_policy(trained_agent, greedy=True)
        trained_success.append(int(success))
        trained_steps.append(steps)

    random_success = []
    random_steps = []
    for _ in range(N_EVAL):
        success, steps = run_policy(random_agent, greedy=False)
        random_success.append(int(success))
        random_steps.append(steps)

    print("=" * 50)
    print(f"Q-learning（贪心策略）: 成功率 {np.mean(trained_success) * 100:5.1f}% | "
          f"平均步数 {np.mean(trained_steps):6.1f}")
    print(f"随机策略对照          : 成功率 {np.mean(random_success) * 100:5.1f}% | "
          f"平均步数 {np.mean(random_steps):6.1f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
