"use strict";

// ============ 网格背景动效 ============
const gridCanvas = document.getElementById("grid-bg");
const gridCtx = gridCanvas.getContext("2d");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const GAP = 34;
let width = 0;
let height = 0;
let drift = 0;

function resizeGrid() {
  width = gridCanvas.width = window.innerWidth;
  height = gridCanvas.height = window.innerHeight;
}

function drawGrid() {
  gridCtx.clearRect(0, 0, width, height);
  gridCtx.strokeStyle = "rgba(130, 180, 205, 0.10)";
  gridCtx.lineWidth = 1;

  const offset = reduceMotion ? 0 : drift % GAP;
  for (let x = offset; x <= width; x += GAP) {
    gridCtx.beginPath();
    gridCtx.moveTo(x, 0);
    gridCtx.lineTo(x, height);
    gridCtx.stroke();
  }
  for (let y = -GAP + offset; y <= height; y += GAP) {
    gridCtx.beginPath();
    gridCtx.moveTo(0, y);
    gridCtx.lineTo(width, y);
    gridCtx.stroke();
  }

  gridCtx.fillStyle = "rgba(45, 212, 191, 0.30)";
  for (let x = offset; x <= width; x += GAP) {
    for (let y = -GAP + offset; y <= height; y += GAP) {
      gridCtx.fillRect(x - 1, y - 1, 2, 2);
    }
  }
}

function gridLoop() {
  if (!reduceMotion) {
    drift = (drift + 0.25) % GAP;
    drawGrid();
  }
  requestAnimationFrame(gridLoop);
}

window.addEventListener("resize", () => {
  resizeGrid();
  drawGrid();
});

resizeGrid();
drawGrid();
if (!reduceMotion) requestAnimationFrame(gridLoop);

// ============ 关键数据滚动计数 ============
const counters = document.querySelectorAll(".hero-stats b");

function finishCount(el) {
  el.textContent = el.dataset.count + (el.dataset.suffix || "");
}

function animateCount(el) {
  const target = parseFloat(el.dataset.count);
  const suffix = el.dataset.suffix || "";
  const start = performance.now();
  const duration = 900;

  function step(now) {
    const progress = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(target * eased) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }

  requestAnimationFrame(step);
}

if (reduceMotion) {
  counters.forEach(finishCount);
} else if ("IntersectionObserver" in window) {
  const counterObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        animateCount(entry.target);
        counterObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.4 });
  counters.forEach((el) => counterObserver.observe(el));
} else {
  counters.forEach(animateCount);
}

// ============ 滚动出现效果 ============
const reveals = document.querySelectorAll(".reveal");

if ("IntersectionObserver" in window) {
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });
  reveals.forEach((el) => revealObserver.observe(el));
} else {
  reveals.forEach((el) => el.classList.add("visible"));
}
