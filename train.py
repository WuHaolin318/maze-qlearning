"""
train.py —— 训练主程序

流程：
1. 创建环境 GridMazeEnv 和智能体 QLearningAgent；
2. 反复训练 N_EPISODES 局（episode），每局从起点出发，直到到达终点或超过最大步数；
3. 每一步调用 agent.update() 把经验写回 Q 表；
4. 每局结束后记录总奖励、步数、是否成功，并让探索概率 epsilon 衰减；
5. 画训练曲线并保存 Q 表，供 evaluate.py / visualize.py 使用。

运行：python train.py
输出：results/training_curves.png、results/q_table.npy、results/training_metrics.npz
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 使用无界面后端：只保存图片，不弹窗
import matplotlib.pyplot as plt

from env import GridMazeEnv
from agent import QLearningAgent

N_EPISODES = 3000   # 训练局数
MAX_STEPS = 150     # 每局最多允许走的步数
SEED = 42           # 固定随机种子，保证每次结果可复现
RESULTS_DIR = "results"


def setup_chinese_font():
    """让 matplotlib 能正常显示中文标题。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False


def moving_average(values, window=100):
    """滑动平均：用最近 window 个数的平均值画曲线，让曲线更平滑。"""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i + 1]) / (i - start + 1))
    return result


def train_agent(env, agent, n_episodes, max_steps=MAX_STEPS, verbose=True):
    """训练一个智能体。

    返回三个数组：每局总奖励、每局步数、每局是否成功。
    这段循环就是 Q-learning 的核心“交互-学习”过程：
    智能体选动作 -> 环境反馈 -> 更新 Q 表 -> 直到本局结束。
    """
    episode_rewards = []
    episode_steps = []
    episode_success = []

    for episode in range(1, n_episodes + 1):
        state = env.reset()          # 新的一局，从起点开始
        total_reward = 0.0
        success = False

        for step in range(max_steps):
            action = agent.choose_action(state)          # 智能体选择动作
            next_state, reward, done = env.step(action)  # 环境执行并反馈
            agent.update(state, action, reward, next_state, done)  # 学习
            state = next_state
            total_reward += reward

            if done:                  # 本局结束
                success = state == env.goal_pos
                break

        # 记录本局统计信息
        episode_rewards.append(total_reward)
        episode_steps.append(step + 1)
        episode_success.append(1 if success else 0)

        # 每局结束后降低探索率：前期多探索，后期多利用
        agent.decay_epsilon()

        if verbose and episode % 500 == 0:
            recent_success = episode_success[-100:]
            recent_steps = episode_steps[-100:]
            print(f"episode {episode:>4} | 最近100局成功率 {sum(recent_success):3d}% | "
                  f"平均步数 {np.mean(recent_steps):.1f} | epsilon {agent.epsilon:.3f}")

    return (np.array(episode_rewards),
            np.array(episode_steps),
            np.array(episode_success))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    env = GridMazeEnv(seed=SEED)
    agent = QLearningAgent(
        n_rows=env.rows,
        n_cols=env.cols,
        n_actions=len(env.ACTIONS),
        seed=SEED,
    )

    rewards, steps, success = train_agent(env, agent, N_EPISODES)

    # 保存训练成果
    np.save(os.path.join(RESULTS_DIR, "q_table.npy"), agent.q_table)
    np.savez(os.path.join(RESULTS_DIR, "training_metrics.npz"),
             rewards=rewards, steps=steps, success=success)

    # 画图：左图是奖励曲线，右图是成功率曲线
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(moving_average(rewards), label="平均总奖励")
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("总奖励")
    axes[0].set_title("训练奖励曲线（滑动平均 100 局）")
    axes[0].legend()

    axes[1].plot(moving_average(success), label="成功率", color="green")
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("成功率")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("训练成功率曲线（滑动平均 100 局）")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150)

    print("训练完成！")
    print(f"最后 100 局平均奖励：{np.mean(rewards[-100:]):.2f}")
    print(f"最后 100 局成功率：{np.mean(success[-100:]) * 100:.1f}%")
    print(f"成果文件：results/training_curves.png, results/q_table.npy")


if __name__ == "__main__":
    main()
