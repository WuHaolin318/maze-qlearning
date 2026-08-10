"use strict";

// ============ 环境定义（与 Python 版 env.py 保持一致） ============
// 0 = 空地，1 = 墙，2 = 起点，3 = 终点
const MAP = [
  [2, 0, 1, 0, 0, 0],
  [0, 0, 1, 0, 1, 0],
  [0, 0, 0, 0, 1, 0],
  [1, 0, 0, 0, 0, 0],
  [1, 1, 1, 0, 0, 0],
  [0, 0, 0, 0, 0, 3],
];
const ROWS = MAP.length;
const COLS = MAP[0].length;
const ACTIONS = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // 上、下、左、右
const ACTION_NAMES = ["上", "下", "左", "右"];
const REWARD_GOAL = 10.0;
const REWARD_WALL = -1.0;
const REWARD_STEP = -0.1;
const MAX_STEPS = 150;
const MA_WINDOW = 100;

function findCell(value) {
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (MAP[r][c] === value) return [r, c];
    }
  }
  throw new Error("地图中找不到编码为 " + value + " 的格子");
}

const START = findCell(2);
const GOAL = findCell(3);

function stateIndex(state) {
  return state[0] * COLS + state[1];
}

// ============ 学习状态 ============
let q = new Float32Array(ROWS * COLS * 4); // Q 表：36 个状态 × 4 个动作
let agentPos = [START[0], START[1]];
let episodePath = [agentPos.slice()];
let episodeReward = 0;
let stepCount = 0;
let done = false;
let episode = 0;
let epsilon = 1.0;
let rewards = [];    // 每局总奖励
let stepsList = [];  // 每局步数
let successes = [];  // 每局是否成功
let lastUpdate = null;
let playing = false;
let rafId = null;

// ============ DOM 元素 ============
const mazeCanvas = document.getElementById("maze");
const mazeCtx = mazeCanvas.getContext("2d");
const curvesCanvas = document.getElementById("curves");
const curvesCtx = curvesCanvas.getContext("2d");
const playBtn = document.getElementById("playBtn");
const playIcon = document.getElementById("playIcon");
const playLabel = document.getElementById("playLabel");
const stepBtn = document.getElementById("stepBtn");
const resetBtn = document.getElementById("resetBtn");
const statEpisode = document.getElementById("statEpisode");
const statEpsilon = document.getElementById("statEpsilon");
const statSuccess = document.getElementById("statSuccess");
const statSteps = document.getElementById("statSteps");
const lastLine = document.getElementById("lastLine");
const formulaLine = document.getElementById("formulaLine");

function cssVar(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

// ============ 超参数读取 ============
function alpha() {
  return parseFloat(document.getElementById("alpha").value);
}

function gamma() {
  return parseFloat(document.getElementById("gamma").value);
}

function epsilonStart() {
  return parseFloat(document.getElementById("epsilonStart").value);
}

function epsilonDecay() {
  return parseFloat(document.getElementById("epsilonDecay").value);
}

// ============ 动作选择：epsilon-greedy ============
function chooseAction(state) {
  const idx = stateIndex(state) * 4;
  if (Math.random() < epsilon) {
    return Math.floor(Math.random() * 4);
  }
  let best = 0;
  for (let a = 1; a < 4; a++) {
    if (q[idx + a] > q[idx + best]) best = a;
  }
  return best;
}

function maxQFor(r, c) {
  const idx = (r * COLS + c) * 4;
  let m = 0;
  for (let a = 0; a < 4; a++) m = Math.max(m, q[idx + a]);
  return m;
}

function bestAction(r, c) {
  const idx = (r * COLS + c) * 4;
  let best = 0;
  for (let a = 1; a < 4; a++) {
    if (q[idx + a] > q[idx + best]) best = a;
  }
  return best;
}

// ============ Q 更新公式 ============
function qUpdate(s, action, reward, ns, isDone) {
  const sIdx = stateIndex(s) * 4;
  const nsIdx = stateIndex(ns) * 4;
  let bestNext = 0;
  if (!isDone) {
    for (let a = 0; a < 4; a++) bestNext = Math.max(bestNext, q[nsIdx + a]);
  }
  const oldQ = q[sIdx + action];
  const target = reward + gamma() * bestNext;
  const newQ = oldQ + alpha() * (target - oldQ);
  q[sIdx + action] = newQ;
  lastUpdate = {
    state: s.slice(),
    actionName: ACTION_NAMES[action],
    reward,
    oldQ,
    target,
    bestNext,
    newQ,
    done: isDone,
  };
}

// ============ 一局的生命周期 ============
function startEpisode() {
  agentPos = [START[0], START[1]];
  episodePath = [agentPos.slice()];
  episodeReward = 0;
  stepCount = 0;
  done = false;
}

function endEpisode(success) {
  rewards.push(episodeReward);
  stepsList.push(stepCount);
  successes.push(success ? 1 : 0);
  episode++;
  epsilon = Math.max(0.05, epsilon * epsilonDecay());
  startEpisode();
}

function doOneAction() {
  if (done) {
    startEpisode();
    return;
  }
  const action = chooseAction(agentPos);
  const [r, c] = agentPos;
  const [dr, dc] = ACTIONS[action];
  const nr = r + dr;
  const nc = c + dc;

  let next;
  let reward;
  let isDone = false;
  if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS || MAP[nr][nc] === 1) {
    next = [r, c];
    reward = REWARD_WALL;
  } else {
    next = [nr, nc];
    if (MAP[nr][nc] === 3) {
      reward = REWARD_GOAL;
      isDone = true;
    } else {
      reward = REWARD_STEP;
    }
  }

  qUpdate(agentPos, action, reward, next, isDone);
  agentPos = next;
  stepCount++;
  episodeReward += reward;
  episodePath.push(next.slice());

  if (isDone) {
    endEpisode(true);
  } else if (stepCount >= MAX_STEPS) {
    endEpisode(false);
  }
}

// ============ 绘制迷宫 ============
function drawMaze() {
  const size = mazeCanvas.width;
  const cell = size / ROWS;
  const ctx = mazeCtx;
  ctx.clearRect(0, 0, size, size);

  const wallColor = cssVar("--wall", "#3f3f46");
  const freeColor = cssVar("--free", "#fafaf9");
  const lineColor = cssVar("--line", "#d9d6d0");
  const textColor = cssVar("--ink", "#1f2328");
  const showQ = document.getElementById("showQ").checked;
  const showPolicy = document.getElementById("showPolicy").checked;
  let maxQ = 0;
  for (let i = 0; i < q.length; i++) maxQ = Math.max(maxQ, q[i]);

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const x = c * cell;
      const y = r * cell;
      ctx.fillStyle = MAP[r][c] === 1 ? wallColor : freeColor;
      ctx.fillRect(x, y, cell, cell);

      if (MAP[r][c] !== 1 && showQ && maxQ > 0) {
        const ratio = maxQFor(r, c) / maxQ;
        ctx.fillStyle = "rgba(15, 118, 110, " + (0.05 + 0.30 * ratio).toFixed(3) + ")";
        ctx.fillRect(x, y, cell, cell);
      }

      ctx.strokeStyle = lineColor;
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
    }
  }

  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "bold 24px sans-serif";
  ctx.fillStyle = cssVar("--blue", "#2563eb");
  ctx.fillText("S", START[1] * cell + cell / 2, START[0] * cell + cell / 2);
  ctx.fillStyle = cssVar("--green", "#16a34a");
  ctx.fillText("G", GOAL[1] * cell + cell / 2, GOAL[0] * cell + cell / 2);

  if (showPolicy) {
    const arrows = ["↑", "↓", "←", "→"];
    ctx.font = "bold 22px sans-serif";
    ctx.fillStyle = textColor;
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        if (MAP[r][c] === 1) continue;
        if ((r === START[0] && c === START[1]) || (r === GOAL[0] && c === GOAL[1])) continue;
        ctx.fillText(arrows[bestAction(r, c)], c * cell + cell / 2, r * cell + cell / 2);
      }
    }
  }

  if (episodePath.length > 1) {
    ctx.strokeStyle = "rgba(234, 88, 12, 0.55)";
    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    ctx.beginPath();
    episodePath.forEach((p, i) => {
      const px = p[1] * cell + cell / 2;
      const py = p[0] * cell + cell / 2;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }

  ctx.fillStyle = cssVar("--orange", "#ea580c");
  ctx.beginPath();
  ctx.arc(agentPos[1] * cell + cell / 2, agentPos[0] * cell + cell / 2, cell * 0.28, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.stroke();
}

// ============ 绘制训练曲线 ============
function movingAverage(values) {
  const out = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= MA_WINDOW) sum -= values[i - MA_WINDOW];
    out.push(sum / Math.min(i + 1, MA_WINDOW));
  }
  return out;
}

function drawPolyline(points, color) {
  const ctx = curvesCtx;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach(([x, y], i) => {
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawCurves() {
  const w = curvesCanvas.width;
  const h = curvesCanvas.height;
  const ctx = curvesCtx;
  ctx.clearRect(0, 0, w, h);

  const padL = 46;
  const padR = 46;
  const padT = 16;
  const padB = 28;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const lineColor = cssVar("--line", "#d9d6d0");
  const textColor = cssVar("--ink", "#1f2328");

  ctx.font = "12px sans-serif";
  ctx.fillStyle = textColor;
  if (rewards.length < 2) {
    ctx.textAlign = "center";
    ctx.fillText("训练开始后显示奖励与成功率曲线", w / 2, h / 2);
    return;
  }

  const rewardMA = movingAverage(rewards);
  const successMA = movingAverage(successes);
  const rMin = Math.min(0, ...rewardMA) - 1;
  const rMax = Math.max(1, ...rewardMA) + 1;
  const xFor = (i) => padL + (i / (rewards.length - 1)) * plotW;
  const yReward = (v) => padT + (1 - (v - rMin) / (rMax - rMin)) * plotH;
  const ySuccess = (v) => padT + (1 - v / 100) * plotH;

  ctx.strokeStyle = lineColor;
  ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, plotW, plotH);

  ctx.textAlign = "center";
  ctx.fillStyle = textColor;
  for (let t = 0; t <= 4; t++) {
    const i = Math.round((t * (rewards.length - 1)) / 4);
    ctx.fillText(String(i + 1), xFor(i), h - 12);
  }
  ctx.fillText("episode", padL + plotW / 2, h - 2);

  ctx.textAlign = "right";
  for (let t = 0; t <= 2; t++) {
    const v = rMin + ((rMax - rMin) * t) / 2;
    ctx.fillText(v.toFixed(1), padL - 6, yReward(v) + 4);
  }

  ctx.textAlign = "left";
  for (const v of [0, 50, 100]) {
    ctx.fillText(v + "%", padL + plotW + 6, ySuccess(v) + 4);
  }

  drawPolyline(
    rewardMA.map((v, i) => [xFor(i), yReward(v)]),
    cssVar("--blue", "#2563eb")
  );
  drawPolyline(
    successMA.map((v, i) => [xFor(i), ySuccess(v * 100)]),
    cssVar("--green", "#16a34a")
  );

  ctx.textAlign = "left";
  ctx.fillStyle = cssVar("--blue", "#2563eb");
  ctx.fillRect(padL + 4, padT + 8, 14, 3);
  ctx.fillStyle = textColor;
  ctx.fillText("奖励(滑动平均)", padL + 22, padT + 13);
  ctx.fillStyle = cssVar("--green", "#16a34a");
  ctx.fillRect(padL + 170, padT + 8, 14, 3);
  ctx.fillStyle = textColor;
  ctx.fillText("成功率", padL + 188, padT + 13);
}

// ============ 统计与公式 ============
function recentMean(arr) {
  if (!arr.length) return null;
  const tail = arr.slice(-MA_WINDOW);
  return tail.reduce((a, b) => a + b, 0) / tail.length;
}

function updateStats() {
  statEpisode.textContent = episode;
  statEpsilon.textContent = epsilon.toFixed(3);
  const sr = recentMean(successes);
  statSuccess.textContent = sr === null ? "-" : (sr * 100).toFixed(1) + "%";
  const avgSteps = recentMean(stepsList);
  statSteps.textContent = avgSteps === null ? "-" : avgSteps.toFixed(1);
}

function updateFormula() {
  if (!lastUpdate) {
    lastLine.textContent = "状态 -- · 动作 -- · 奖励 -- · 结束 --";
    formulaLine.textContent = "Q(s,a) ← --";
    return;
  }
  const u = lastUpdate;
  lastLine.textContent =
    "状态 (" + u.state[0] + "," + u.state[1] + ") · 动作 " + u.actionName +
    " · 奖励 " + u.reward.toFixed(2) + " · 结束 " + (u.done ? "是" : "否");
  formulaLine.innerHTML =
    "Q(s,a) ← <b>" + u.oldQ.toFixed(2) + "</b> + " + alpha().toFixed(2) +
    " × ( " + u.reward.toFixed(2) + " + " + gamma().toFixed(2) + " × " +
    u.bestNext.toFixed(2) + " − " + u.oldQ.toFixed(2) + " ) = <b>" +
    u.newQ.toFixed(2) + "</b>";
}

function draw() {
  drawMaze();
  drawCurves();
  updateStats();
  updateFormula();
}

// ============ 播放控制 ============
function frame() {
  if (!playing) return;
  const speed = parseInt(document.getElementById("speed").value, 10);
  for (let i = 0; i < speed; i++) doOneAction();
  draw();
  rafId = requestAnimationFrame(frame);
}

function togglePlay() {
  playing = !playing;
  playLabel.textContent = playing ? "暂停" : "播放";
  playIcon.textContent = playing ? "⏸" : "▶";
  if (playing) {
    rafId = requestAnimationFrame(frame);
  } else if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
}

function stepOnce() {
  doOneAction();
  draw();
}

function resetAll() {
  q = new Float32Array(ROWS * COLS * 4);
  rewards = [];
  stepsList = [];
  successes = [];
  episode = 0;
  epsilon = epsilonStart();
  lastUpdate = null;
  startEpisode();
  draw();
}

// ============ 事件绑定与初始化 ============
playBtn.addEventListener("click", togglePlay);
stepBtn.addEventListener("click", stepOnce);
resetBtn.addEventListener("click", resetAll);

const decimals = { alpha: 2, gamma: 2, epsilonStart: 2, epsilonDecay: 3, speed: 0 };
for (const id of Object.keys(decimals)) {
  const input = document.getElementById(id);
  const display = document.getElementById(id + "Val");
  input.addEventListener("input", () => {
    display.textContent = parseFloat(input.value).toFixed(decimals[id]);
  });
}

resetAll();
