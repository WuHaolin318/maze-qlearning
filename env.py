"""
env.py —— 迷宫环境模块（Environment）

这个模块负责三件事：
1. 用一张二维地图描述迷宫（空地、墙、起点、终点）；
2. 记录智能体当前在哪一格，这一格就是状态 state；
3. 当智能体执行一个动作后，返回“新状态 + 奖励 + 是否结束”。

它和别的模块的关系：
- train.py 会调用 env.reset() 开始一局，然后反复调用 env.step(action)；
- agent.py 只需要知道状态(state)和奖励(reward)，不需要关心地图细节。

核心概念：
- state（状态）：智能体所在的位置，用 (row, col) 表示；
- action（动作）：上、下、左、右，用数字 0、1、2、3 表示；
- reward（奖励）：环境给智能体的反馈分数；
- done（结束）：True 表示本局结束（本项目中表示到达终点）。
"""


class GridMazeEnv:
    """6x6 网格迷宫环境。"""

    # 动作常量：动作编号 -> (行变化量, 列变化量)
    # 0=上(行-1)，1=下(行+1)，2=左(列-1)，3=右(列+1)
    ACTIONS = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }

    # 地图编码：
    # 0 = 空地，1 = 墙（不能进入），2 = 起点，3 = 终点
    # 这个地图由 8 个墙格组成简单通道，保证至少有一条可行路径。
    MAP = [
        [2, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 3],
    ]

    # 奖励设计（为什么这样设计，README 里有解释）：
    REWARD_GOAL = 10.0   # 到达终点：大的正奖励
    REWARD_WALL = -1.0   # 撞墙或越界：惩罚，并原地停留
    REWARD_STEP = -0.1   # 普通移动：小惩罚，引导智能体走更短的路

    def __init__(self, seed=None):
        # 本版本环境是确定性的，seed 参数保留是为了以后扩展随机环境
        self.seed = seed
        self.rows = len(self.MAP)
        self.cols = len(self.MAP[0])
        self.start_pos = self._find_position(2)   # 起点
        self.goal_pos = self._find_position(3)    # 终点
        self.agent_pos = None
        self.reset()   # 创建环境后立即把智能体放回起点

    def _find_position(self, value):
        """在地图里找到编码为 value 的格子，返回 (row, col)。"""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.MAP[r][c] == value:
                    return (r, c)
        raise ValueError(f"地图中找不到编码为 {value} 的格子")

    def reset(self):
        """开始新的一局：把智能体放回起点，返回初始状态。"""
        self.agent_pos = self.start_pos
        return self.agent_pos

    def _is_wall(self, r, c):
        """判断 (r, c) 是否越界或撞墙。越界也按墙处理，不允许走出地图。"""
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return True
        return self.MAP[r][c] == 1

    def step(self, action):
        """执行动作，返回 (next_state, reward, done)。"""
        dr, dc = self.ACTIONS[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc

        if self._is_wall(nr, nc):
            # 撞墙/越界：位置不变，给惩罚，本局不结束
            return self.agent_pos, self.REWARD_WALL, False

        # 正常移动
        self.agent_pos = (nr, nc)
        if self.agent_pos == self.goal_pos:
            return self.agent_pos, self.REWARD_GOAL, True
        return self.agent_pos, self.REWARD_STEP, False

    def render(self):
        """在终端打印当前迷宫：A=智能体，G=终点，#=墙，.=空地。"""
        for r in range(self.rows):
            line = ""
            for c in range(self.cols):
                if (r, c) == self.agent_pos:
                    line += "A "
                elif self.MAP[r][c] == 1:
                    line += "# "
                elif self.MAP[r][c] == 3:
                    line += "G "
                else:
                    line += ". "
            print(line)
        print()
