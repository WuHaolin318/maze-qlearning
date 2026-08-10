"""
visualize.py —— 可视化程序

读取训练好的 Q 表，生成两张图：
1. policy.png：策略箭头图，每个空地格画一个箭头，指向该格 Q 值最大的动作；
2. path.png：一次成功路径图，让训练好的智能体从起点走一次，画出轨迹。

运行：python visualize.py（需要先运行 python train.py）
输出：results/policy.png、results/path.png
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import GridMazeEnv
from agent import QLearningAgent

RESULTS_DIR = "results"
MAX_STEPS = 150

# 动作编号对应的箭头符号，与 env.ACTIONS 一一对应
ACTION_SYMBOLS = {0: "↑", 1: "↓", 2: "←", 3: "→"}


def setup_chinese_font():
    """让 matplotlib 能正常显示中文标题。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False


def best_action_grid(q_table, env):
    """根据 Q 表生成策略网格：每个非墙格子存一个最优动作编号。"""
    grid = []
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            if env.MAP[r][c] == 1:
                row.append(None)   # 墙没有动作
            else:
                row.append(int(np.argmax(q_table[r * env.cols + c])))
        grid.append(row)
    return grid


def draw_map(ax, env):
    """画迷宫底色：墙用深色，空地用浅色。"""
    for r in range(env.rows):
        for c in range(env.cols):
            if env.MAP[r][c] == 1:
                color = "#4a4a4a"
            else:
                color = "#eef2f7"
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       color=color, zorder=1))


def mark_start_goal(ax, env):
    """在图上标出起点 S 和终点 G。"""
    ax.text(env.start_pos[1], env.start_pos[0], "S",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color="#1f6feb", zorder=4)
    ax.text(env.goal_pos[1], env.goal_pos[0], "G",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color="#2ea043", zorder=4)


def plot_policy(q_table, env):
    """画策略箭头图并保存。"""
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_map(ax, env)

    action_grid = best_action_grid(q_table, env)
    for r in range(env.rows):
        for c in range(env.cols):
            if action_grid[r][c] is None or (r, c) == env.goal_pos:
                continue
            ax.text(c, r, ACTION_SYMBOLS[action_grid[r][c]],
                    ha="center", va="center", fontsize=16, zorder=3)

    mark_start_goal(ax, env)
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_title("Q-learning 学到的策略（箭头 = 每格最优动作）")

    save_path = os.path.join(RESULTS_DIR, "policy.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"策略图已保存: {save_path}")


def run_greedy_path(env, agent):
    """让训练好的智能体从起点走一次，返回路径点列表和是否成功。"""
    state = env.reset()
    path = [state]
    for _ in range(MAX_STEPS):
        action = agent.choose_action(state, greedy=True)
        state, _, done = env.step(action)
        path.append(state)
        if done:
            return path, state == env.goal_pos
    return path, False


def plot_path(env, agent):
    """画一次贪心策略的行走轨迹并保存。"""
    setup_chinese_font()
    path, success = run_greedy_path(env, agent)

    print("智能体走出的路径：")
    print(" -> ".join(f"{p[0]},{p[1]}" for p in path))
    print("是否到达终点:", success)

    fig, ax = plt.subplots(figsize=(6, 6))
    draw_map(ax, env)

    xs = [p[1] for p in path]
    ys = [p[0] for p in path]
    ax.plot(xs, ys, "-o", color="#d1242f", linewidth=2.5, markersize=6, zorder=3)

    mark_start_goal(ax, env)
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_title("训练后智能体的一次行走路径")

    save_path = os.path.join(RESULTS_DIR, "path.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"路径图已保存: {save_path}")


def main():
    q_path = os.path.join(RESULTS_DIR, "q_table.npy")
    if not os.path.exists(q_path):
        raise FileNotFoundError("没有找到 results/q_table.npy，请先运行 python train.py")

    q_table = np.load(q_path)

    env = GridMazeEnv()
    agent = QLearningAgent(env.rows, env.cols, n_actions=len(env.ACTIONS))
    agent.q_table = q_table

    plot_policy(q_table, env)
    plot_path(env, agent)


if __name__ == "__main__":
    main()
