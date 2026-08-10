"""
tests.py —— 自动化测试

用 assert 检查环境规则、Q 更新方向、可复现性是否符合预期。
运行：python tests.py
看到“全部测试通过”就说明基础逻辑正确。
"""

from env import GridMazeEnv
from agent import QLearningAgent


def test_env_start():
    env = GridMazeEnv()
    state = env.reset()
    assert state == env.start_pos, "初始状态应该是起点"
    print("test_env_start 通过")


def test_wall_collision():
    env = GridMazeEnv()
    env.reset()
    # 地图 (0, 2) 是墙，从 (0, 1) 向右走会撞墙
    env.agent_pos = (0, 1)
    next_state, reward, done = env.step(3)  # 动作 3 = 右
    assert next_state == (0, 1), "撞墙后位置不变"
    assert reward == env.REWARD_WALL, "撞墙应得到墙惩罚"
    assert done is False, "撞墙不结束本局"
    print("test_wall_collision 通过")


def test_boundary():
    env = GridMazeEnv()
    env.agent_pos = (0, 0)
    next_state, reward, done = env.step(0)  # 动作 0 = 上，越界
    assert next_state == (0, 0), "越界后位置不变"
    assert reward == env.REWARD_WALL, "越界应得到墙惩罚"
    assert done is False, "越界不结束本局"
    print("test_boundary 通过")


def test_goal():
    env = GridMazeEnv()
    env.agent_pos = (5, 4)
    next_state, reward, done = env.step(3)  # 向右一步到达终点
    assert next_state == env.goal_pos, "应到达终点"
    assert reward == env.REWARD_GOAL, "到达终点应得到 +10"
    assert done is True, "到达终点应结束本局"
    print("test_goal 通过")


def test_normal_step():
    env = GridMazeEnv()
    env.agent_pos = (0, 0)
    next_state, reward, done = env.step(1)  # 动作 1 = 下
    assert next_state == (1, 0), "普通移动应更新位置"
    assert reward == env.REWARD_STEP, "普通移动应得到小惩罚"
    assert done is False, "普通移动不结束本局"
    print("test_normal_step 通过")


def test_q_update_direction():
    agent = QLearningAgent(2, 2, n_actions=4, alpha=0.5, gamma=0.9)
    # 初始 Q=0；奖励 1，下一个状态不是终点（best_next_q=0）
    # TD target = 1 + 0.9 * 0 = 1，所以 Q 应变成 0.5
    agent.update((0, 0), 0, 1.0, (0, 1), done=False)
    assert agent.q_table[0, 0] == 0.5, "Q 值应按公式增大"
    print("test_q_update_direction 通过")


def test_q_update_goal():
    agent = QLearningAgent(2, 2, n_actions=4, alpha=0.5, gamma=0.9)
    # done=True 时只算当前奖励：TD target = 1
    agent.update((0, 0), 1, 1.0, (0, 1), done=True)
    assert agent.q_table[0, 1] == 0.5, "终点后不再估计未来回报"
    print("test_q_update_goal 通过")


def test_reproducible():
    agent1 = QLearningAgent(2, 2, n_actions=4, epsilon=1.0, seed=7)
    agent2 = QLearningAgent(2, 2, n_actions=4, epsilon=1.0, seed=7)
    acts1 = [agent1.choose_action((0, 0)) for _ in range(5)]
    acts2 = [agent2.choose_action((0, 0)) for _ in range(5)]
    assert acts1 == acts2, "相同随机种子应产生相同动作序列"
    print("test_reproducible 通过")


def test_greedy_picks_best():
    agent = QLearningAgent(2, 2, n_actions=4)
    agent.q_table[0, :] = [0.1, 0.5, 0.2, 0.3]
    assert agent.choose_action((0, 0), greedy=True) == 1, "贪心应选 Q 最大的动作"
    print("test_greedy_picks_best 通过")


if __name__ == "__main__":
    test_env_start()
    test_wall_collision()
    test_boundary()
    test_goal()
    test_normal_step()
    test_q_update_direction()
    test_q_update_goal()
    test_reproducible()
    test_greedy_picks_best()
    print("全部测试通过")
