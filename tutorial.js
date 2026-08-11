"use strict";

const steps = Array.from(document.querySelectorAll(".step"));
const stepNav = document.getElementById("stepNav");
const progressFill = document.getElementById("progressFill");
const progressText = document.getElementById("progressText");
const STORAGE_KEY = "maze-qlearning-tutorial";

let current = 0;
let completed = {};

function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (saved) {
      completed = saved.completed || {};
      current = Math.min(steps.length - 1, saved.current || 0);
    }
  } catch (err) {
    completed = {};
    current = 0;
  }
}

function saveState() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ completed, current }));
  } catch (err) {
    // 某些隐私模式下 localStorage 不可用，忽略即可
  }
}

function renderNav() {
  stepNav.innerHTML = "";
  steps.forEach((step, index) => {
    const link = document.createElement("a");
    link.href = "#step-" + (index + 1);
    link.className = "nav-link";
    link.textContent = (index + 1).toString().padStart(2, "0") + " " + step.dataset.title;
    if (index === current) link.classList.add("active");
    if (completed[step.id]) link.classList.add("done");
    link.addEventListener("click", (event) => {
      event.preventDefault();
      showStep(index);
    });
    stepNav.appendChild(link);
  });
}

function showStep(index) {
  current = Math.max(0, Math.min(steps.length - 1, index));
  steps.forEach((step, i) => {
    step.classList.toggle("active", i === current);
  });
  const checkbox = steps[current].querySelector(".step-done");
  if (checkbox) checkbox.checked = Boolean(completed[steps[current].id]);

  renderNav();
  progressFill.style.width = ((current + 1) / steps.length) * 100 + "%";
  progressText.textContent = (current + 1) + " / " + steps.length;
  saveState();
  window.scrollTo({ top: 0, behavior: "auto" });
}

function initCheckboxes() {
  steps.forEach((step) => {
    const checkbox = step.querySelector(".step-done");
    if (!checkbox) return;
    checkbox.addEventListener("change", () => {
      completed[step.id] = checkbox.checked;
      saveState();
      renderNav();
    });
  });
}

function initStepNavButtons() {
  steps.forEach((step, index) => {
    const prev = step.querySelector(".prev-btn");
    const next = step.querySelector(".next-btn");
    if (prev) {
      prev.disabled = index === 0;
      prev.addEventListener("click", () => showStep(index - 1));
    }
    if (next) {
      next.addEventListener("click", () => showStep(index + 1));
    }
  });
}

function initCopyButtons() {
  document.querySelectorAll(".copy-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const textarea = button.nextElementSibling;
      const text = textarea.value;
      try {
        await navigator.clipboard.writeText(text);
      } catch (err) {
        textarea.focus();
        textarea.select();
        document.execCommand("copy");
      }
      const original = button.textContent;
      button.textContent = "已复制";
      setTimeout(() => {
        button.textContent = original;
      }, 1500);
    });
  });
}

loadState();
initCheckboxes();
initStepNavButtons();
initCopyButtons();
showStep(current);
