"""make_promo.py —— 生成项目宣传视频（1280x720 MP4）"""

import os
import shutil
import subprocess
import wave
from functools import partial

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 720
FPS = 30
DURATIONS = [3.0, 3.0, 2.4, 2.4, 2.4, 3.0]

MEDIA_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(MEDIA_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FRAMES_DIR = os.path.join(MEDIA_DIR, "frames")
OUTPUT = os.path.join(MEDIA_DIR, "maze-qlearning-promo.mp4")
OUTPUT_SILENT = os.path.join(MEDIA_DIR, "maze-qlearning-promo-silent.mp4")
MUSIC = os.path.join(MEDIA_DIR, "maze-qlearning-music.wav")

BG = (7, 11, 16)
TEAL = (45, 212, 191)
INK = (233, 238, 243)
MUTED = (154, 168, 180)
BLUE = (90, 162, 255)

FONT_CACHE = {}


def load_font(size, bold=False):
    """加载中文字体并缓存，避免每帧重复读取。"""
    key = (size, bold)
    if key in FONT_CACHE:
        return FONT_CACHE[key]
    bold_paths = ["C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/simhei.ttf"]
    regular_paths = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
    candidates = bold_paths if bold else regular_paths
    for path in candidates:
        if os.path.exists(path):
            try:
                FONT_CACHE[key] = ImageFont.truetype(path, size)
                return FONT_CACHE[key]
            except OSError:
                continue
    FONT_CACHE[key] = ImageFont.load_default()
    return FONT_CACHE[key]


def smoothstep(t):
    return t * t * (3 - 2 * t)


def clip(t):
    return max(0.0, min(1.0, t))


def new_frame():
    return Image.new("RGBA", (WIDTH, HEIGHT), BG + (255,))


def draw_bg(draw):
    """画科技感网格背景。"""
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=BG + (255,))
    step = 48
    for x in range(0, WIDTH + 1, step):
        draw.line([(x, 0), (x, HEIGHT)], fill=(45, 212, 191, 16), width=1)
    for y in range(0, HEIGHT + 1, step):
        draw.line([(0, y), (WIDTH, y)], fill=(45, 212, 191, 16), width=1)
    draw.line([(0, HEIGHT - 110), (WIDTH, 110)], fill=(45, 212, 191, 36), width=2)


def text_size(draw, text, f):
    bbox = draw.textbbox((0, 0), text, font=f)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def centered(draw, cx, cy, text, f, fill):
    w, h = text_size(draw, text, f)
    draw.text((cx - w / 2, cy - h / 2), text, font=f, fill=fill)


def paste_fade(base, img, pos, alpha):
    """把图片按透明度合成到背景上。"""
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.alpha_composite(img, pos)
    alpha_channel = layer.getchannel("A").point(lambda v: int(v * alpha))
    layer.putalpha(alpha_channel)
    base.alpha_composite(layer)


def load_scaled(path, max_w, max_h):
    img = Image.open(path).convert("RGBA")
    scale = min(max_w / img.width, max_h / img.height)
    return img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)


def load_crop(path, w, h):
    """把网页截图裁剪成固定比例并缩放，让画面填满内容区。"""
    img = Image.open(path).convert("RGB")
    target = w / h
    if img.width / img.height > target:
        new_w = int(img.height * target)
        left = (img.width - new_w) // 2
        img = img.crop((left, 0, left + new_w, img.height))
    else:
        new_h = int(img.width / target)
        top = (img.height - new_h) // 2
        img = img.crop((0, top, img.width, top + new_h))
    return img.convert("RGBA").resize((w, h), Image.LANCZOS)


def add_progress(img, frac):
    draw = ImageDraw.Draw(img)
    y = HEIGHT - 18
    draw.rectangle([0, y, WIDTH, y + 3], fill=(148, 184, 208, 40))
    draw.rectangle([0, y, int(WIDTH * frac), y + 3], fill=TEAL + (230,))


def scene_title(t):
    img = new_frame()
    draw = ImageDraw.Draw(img)
    draw_bg(draw)
    ease = smoothstep(clip(t / 0.6))
    offset = int((1 - ease) * 26)

    centered(draw, WIDTH / 2, 195 + offset, "Q-learning",
             load_font(78, True), TEAL + (int(255 * ease),))
    centered(draw, WIDTH / 2, 300 + offset, "迷宫自主导航系统",
             load_font(58, True), INK + (int(255 * ease),))
    centered(draw, WIDTH / 2, 415, "强化学习 · 手把手教学 · 从零实现",
             load_font(28), MUTED + (255,))
    centered(draw, WIDTH / 2, 470, "一个函数一课，带你从零写出完整项目",
             load_font(26), INK + (255,))
    centered(draw, WIDTH / 2, 585, "WuHaolin318.github.io/maze-qlearning",
             load_font(24), BLUE + (255,))
    return img


def scene_image(t, path, caption, note):
    img = new_frame()
    draw = ImageDraw.Draw(img)
    draw_bg(draw)
    ease = smoothstep(clip(t / 0.5))

    content = load_scaled(path, 900, 430)
    x = int((WIDTH - content.width) / 2 - (1 - ease) * 80)
    y = 150
    paste_fade(img, content, (x, y), ease)
    draw.rectangle(
        [x - 6, y - 6, x + content.width + 6, y + content.height + 6],
        outline=TEAL + (int(140 * ease),),
        width=3,
    )
    centered(draw, WIDTH / 2, 640, caption, load_font(36, True), INK + (255,))
    centered(draw, WIDTH / 2, 688, note, load_font(22), MUTED + (255,))
    return img


def scene_teaching(t, path, caption, note):
    """网页截图场景：强调手把手跟着网页写代码。"""
    img = new_frame()
    draw = ImageDraw.Draw(img)
    draw_bg(draw)
    ease = smoothstep(clip(t / 0.5))

    content = load_crop(path, 980, 470)
    x = int((WIDTH - content.width) / 2 - (1 - ease) * 80)
    y = 125
    paste_fade(img, content, (x, y), ease)
    draw.rectangle(
        [x - 6, y - 6, x + content.width + 6, y + content.height + 6],
        outline=TEAL + (int(140 * ease),),
        width=3,
    )
    centered(draw, WIDTH / 2, 635, caption, load_font(36, True), INK + (255,))
    centered(draw, WIDTH / 2, 685, note, load_font(22), MUTED + (255,))
    return img


def scene_end(t):
    img = new_frame()
    draw = ImageDraw.Draw(img)
    draw_bg(draw)
    ease = smoothstep(clip(t / 0.6))

    centered(draw, WIDTH / 2, 260, "开始你的强化学习之旅",
             load_font(62, True), INK + (int(255 * ease),))
    centered(draw, WIDTH / 2, 350,
             "跟着 25 步函数与参数教程，从零亲手写出来",
             load_font(28), MUTED + (int(255 * ease),))
    centered(draw, WIDTH / 2, 455,
             "WuHaolin318.github.io/maze-qlearning",
             load_font(28), TEAL + (int(255 * ease),))

    chips = ["教程", "学习台", "GitHub"]
    chip_w, chip_h, gap = 150, 52, 26
    total_w = len(chips) * chip_w + (len(chips) - 1) * gap
    start_x = (WIDTH - total_w) / 2
    for i, label in enumerate(chips):
        cx = start_x + i * (chip_w + gap)
        cy = 560
        draw.rounded_rectangle([cx, cy, cx + chip_w, cy + chip_h],
                               radius=8, outline=TEAL + (int(160 * ease),), width=2)
        centered(draw, cx + chip_w / 2, cy + chip_h / 2, label,
                 load_font(22, True), INK + (int(255 * ease),))
    return img


def render_frames():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    total_frames = int(sum(DURATIONS) * FPS)
    scenes = [
        scene_title,
        partial(scene_teaching,
                path=os.path.join(MEDIA_DIR, "shot_tutorial.png"),
                caption="手把手教学：一个函数一课",
                note="25 个函数与参数步骤，带你逐步写代码"),
        partial(scene_image, path=os.path.join(RESULTS_DIR, "policy.png"),
                caption="100% 成功率 · 10 步最优路径",
                note="从完全乱走，到学会每一步该往哪走"),
        partial(scene_image, path=os.path.join(RESULTS_DIR, "training_curves.png"),
                caption="3000 局训练曲线",
                note="奖励稳步上升，成功率收敛到 100%"),
        partial(scene_teaching,
                path=os.path.join(MEDIA_DIR, "shot_demo.png"),
                caption="交互式学习台",
                note="实时调参，单步观察每一次 Q 更新"),
        scene_end,
    ]

    for frame_index in range(total_frames):
        t = frame_index / FPS
        acc = 0.0
        frame = None
        for si, duration in enumerate(DURATIONS):
            if t < acc + duration:
                frame = scenes[si](t - acc)
                break
            acc += duration
        if frame is None:
            frame = scenes[-1](DURATIONS[-1])
        add_progress(frame, (frame_index + 1) / total_frames)
        frame.convert("RGB").save(
            os.path.join(FRAMES_DIR, "frame_%04d.png" % (frame_index + 1))
        )


def encode_silent():
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", os.path.join(FRAMES_DIR, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        "-movflags", "+faststart",
        OUTPUT_SILENT,
    ]
    subprocess.run(cmd, check=True)


def make_music(path, duration):
    """生成一段轻柔的合成背景音乐（WAV）。"""
    sample_rate = 44100
    n = int(sample_rate * duration)
    stereo = np.zeros((n, 2))
    chords = [
        (220.00, 261.63, 329.63),
        (174.61, 220.00, 261.63),
        (130.81, 164.81, 196.00),
        (196.00, 246.94, 293.66),
    ]
    bass = [110.00, 87.31, 65.41, 98.00]
    chord_dur = duration / len(chords)

    def add_note(freq, start, dur, amp):
        i0 = int(start * sample_rate)
        i1 = min(n, int((start + dur) * sample_rate))
        if i1 <= i0:
            return
        tt = np.arange(i1 - i0) / sample_rate
        envelope = np.minimum(tt / 0.03, 1.0) * np.exp(-tt * 2.4)
        tone = (
            np.sin(2 * np.pi * freq * tt)
            + 0.35 * np.sin(2 * np.pi * freq * 2 * tt)
            + 0.10 * np.sin(2 * np.pi * freq * 3 * tt)
        )
        tone = tone * envelope * amp
        pan = 0.5 + 0.12 * np.sin(start * 2.1)
        stereo[i0:i1, 0] += tone * (1 - pan)
        stereo[i0:i1, 1] += tone * pan

    for ci, (chord, bass_freq) in enumerate(zip(chords, bass)):
        cstart = ci * chord_dur
        notes = chord + chord
        step = chord_dur / len(notes)
        for j, freq in enumerate(notes):
            add_note(freq, cstart + j * step, step * 0.85, 0.14)
        add_note(bass_freq, cstart, chord_dur, 0.11)

    delay = int(0.32 * sample_rate)
    stereo = stereo + 0.18 * np.roll(stereo, delay, axis=0)
    stereo[:delay] *= 0.0
    peak = max(1e-9, float(np.abs(stereo).max()))
    stereo = stereo / peak * 0.5
    pcm = (stereo * 32767).astype(np.int16)

    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def mux_audio():
    cmd = [
        "ffmpeg", "-y",
        "-i", OUTPUT_SILENT,
        "-i", MUSIC,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        OUTPUT,
    ]
    subprocess.run(cmd, check=True)


def main():
    total_duration = sum(DURATIONS)
    render_frames()
    encode_silent()
    make_music(MUSIC, total_duration)
    mux_audio()
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    if os.path.exists(OUTPUT_SILENT):
        os.remove(OUTPUT_SILENT)
    print("视频已生成:", OUTPUT)


if __name__ == "__main__":
    main()
