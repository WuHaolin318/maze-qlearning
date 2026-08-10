"""
experiments.py —— 实验分析程序

做三组单变量实验（一次只改一个超参数，其他保持不变）：
1. 学习率 alpha：0.05 / 0.1 / 0.3，比较学习速度与最终表现；
2. 训练轮数：500 / 1500 / 3000，比较训练量是否足够；
3. 探索策略：固定 epsilon 0.05 / 0.3 与衰减 1.0→0.05 对比。

每组都记录“首次达到 90% 成功率”的位置（学习速度指标），
训练结束后再用贪心策略评估 100 局，记录最终成功率与平均步数。

运行：python experiments.py
输出：results/alpha_experiment.png、results/episodes_experiment.png、
      results/epsilon_experiment.png
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import GridMazeEnv
from agent import QLearningAgent
from train import train_agent, setup_chinese_font, MAX_STEPS, moving_average

RESULTS_DIR = "results"
SEED = 42
N_EVAL = 100


def evaluate(agent, n_eval=N_EVAL):
    """用贪心策略评估 n_eval 局，返回 (成功率, 平均步数)。"""
    env = GridMazeEnv()
    successes = []
    steps = []

    for _ in range(n_eval):
        state = env.reset()
        for step in range(MAX_STEPS):
            action = agent.choose_action(state, greedy=True)
            state, _, done = env.step(action)
            if done:
                break
        successes.append(1 if state == env.goal_pos else 0)
        steps.append(step + 1)

    return float(np.mean(successes)), float(np.mean(steps))


def make_agent(env, alpha=0.1, epsilon=1.0, epsilon_decay=0.999):
    """创建一个使用指定超参数的 Q-learning 智能体。"""
    return QLearningAgent(
        n_rows=env.rows,
        n_cols=env.cols,
        n_actions=len(env.ACTIONS),
        alpha=alpha,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        seed=SEED,
    )


def first_time_above(success, threshold=0.9, window=100):
    """返回连续 window 局成功率第一次达到 threshold 的局数（从 1 开始计数）。

    只用“完整窗口”计算成功率，避免前几局碰巧成功造成的假象。
    """
    success = list(success)
    for i in range(window - 1, len(success)):
        recent = success[i - window + 1:i + 1]
        if sum(recent) / window >= threshold:
            return i + 1
    return None


def run_alpha_experiment():
    """实验 1：改变学习率 alpha，比较学习速度与最终表现。"""
    alphas = [0.05, 0.1, 0.3]
    n_episodes = 1500
    results = []
    curves = []

    print("实验 1：不同学习率 alpha 的影响（训练 1500 局）")
    for alpha in alphas:
        env = GridMazeEnv()
        agent = make_agent(env, alpha=alpha)
        _, _, success = train_agent(env, agent, n_episodes, verbose=False)
        curve = moving_average(success, window=100)
        curves.append(curve)

        first90 = first_time_above(success)
        success_rate, avg_steps = evaluate(agent)
        results.append((alpha, success_rate, avg_steps, first90))

        first_str = f"episode {first90}" if first90 else "未达到"
        print(f"alpha = {alpha:<4} | 首次90%成功率: {first_str:>12} | "
              f"最终成功率 {success_rate * 100:5.1f}% | 平均步数 {avg_steps:6.1f}")

    # 画学习速度对比曲线
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#f1a208", "#2da44e", "#57606a"]
    for (alpha, _, _, first90), curve, color in zip(results, curves, colors):
        label = f"alpha={alpha}"
        if first90:
            label += f"（episode {first90} 达90%）"
        ax.plot(curve, color=color, label=label)
    ax.set_xlabel("episode")
    ax.set_ylabel("成功率（滑动平均100局）")
    ax.set_ylim(0, 1.05)
    ax.set_title("不同学习率 alpha 的学习速度对比")
    ax.legend()
    fig.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "alpha_experiment.png")
    fig.savefig(save_path, dpi=150)
    print(f"实验 1 图表已保存: {save_path}\n")


def run_episodes_experiment():
    """实验 2：改变训练轮数，验证训练量是否足够。"""
    episode_counts = [500, 1500, 3000]
    results = []

    print("实验 2：不同训练轮数的影响（alpha = 0.1）")
    for n in episode_counts:
        env = GridMazeEnv()
        agent = make_agent(env)
        train_agent(env, agent, n, verbose=False)
        success_rate, avg_steps = evaluate(agent)
        results.append((n, success_rate, avg_steps))
        print(f"训练 {n:>4} 局 | 成功率 {success_rate * 100:5.1f}% | "
              f"平均步数 {avg_steps:6.1f}")

    # 画最终成功率对比图
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [n for n, _, _ in results]
    ys = [s * 100 for _, s, _ in results]
    bars = ax.bar([str(n) for n in xs], ys, color=["#f1a208", "#2da44e", "#57606a"])
    for bar, (_, _, avg_steps) in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"平均步数 {avg_steps:.0f}", ha="center", fontsize=10)
    ax.set_xlabel("训练轮数（episodes）")
    ax.set_ylabel("成功率 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("不同训练轮数的最终成功率（alpha = 0.1）")
    fig.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "episodes_experiment.png")
    fig.savefig(save_path, dpi=150)
    print(f"实验 2 图表已保存: {save_path}")
    print("提示：小地图上 500 局已收敛，因此最终成功率相同；"
          "学习速度更适合看训练曲线。\n")


def run_epsilon_experiment():
    """实验 3：比较固定 epsilon 与衰减 epsilon 的学习效果。"""
    strategies = [
        ("固定 epsilon=0.05", 0.05, 1.0),
        ("固定 epsilon=0.30", 0.30, 1.0),
        ("衰减 1.0→0.05", 1.0, 0.999),
    ]
    n_episodes = 1500
    results = []
    curves = []

    print("实验 3：不同探索策略的影响（训练 1500 局）")
    for name, epsilon, decay in strategies:
        env = GridMazeEnv()
        agent = make_agent(env, epsilon=epsilon, epsilon_decay=decay)
        _, _, success = train_agent(env, agent, n_episodes, verbose=False)
        curve = moving_average(success, window=100)
        curves.append(curve)

        first90 = first_time_above(success)
        success_rate, avg_steps = evaluate(agent)
        results.append((name, success_rate, avg_steps, first90))

        first_str = f"episode {first90}" if first90 else "未达到"
        print(f"{name:<16} | 首次90%成功率: {first_str:>12} | "
              f"最终成功率 {success_rate * 100:5.1f}% | 平均步数 {avg_steps:6.1f}")

    # 画学习速度对比曲线
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#f1a208", "#2da44e", "#57606a"]
    for (name, _, _, first90), curve, color in zip(results, curves, colors):
        label = name
        if first90:
            label += f"（episode {first90} 达90%）"
        ax.plot(curve, color=color, label=label)
    ax.set_xlabel("episode")
    ax.set_ylabel("成功率（滑动平均100局）")
    ax.set_ylim(0, 1.05)
    ax.set_title("不同探索策略的学习速度对比")
    ax.legend()
    fig.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "epsilon_experiment.png")
    fig.savefig(save_path, dpi=150)
    print(f"实验 3 图表已保存: {save_path}\n")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_alpha_experiment()
    run_episodes_experiment()
    run_epsilon_experiment()


if __name__ == "__main__":
    main()
