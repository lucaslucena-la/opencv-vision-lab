import cv2
import math
import os
import sys
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageFont
from pilmoji import Pilmoji

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOVER_SECONDS = 2.0

# BGR colors
ITEMS = [
    {"label": "Frutas",  "emoji": "🍎", "file": "fruits.py",   "color": (40, 180,  40)},
    {"label": "Maos",    "emoji": "🖐️",  "file": "hands.py",    "color": (200,  80,   0)},
    {"label": "Pinca",   "emoji": "🍪", "file": "pinch.py",    "color": (0,  130, 200)},
    {"label": "Jogo",    "emoji": "🎮", "file": "handgame.py", "color": (180,   0, 180)},
]

# ── MediaPipe ──────────────────────────────────────────────────────────────────
BaseOptions          = python.BaseOptions
HandLandmarker       = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode    = vision.RunningMode

landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=os.path.join(SCRIPT_DIR, "hand_landmarker.task")
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
))

# ── Camera ─────────────────────────────────────────────────────────────────────
def open_camera(indices=(0, 1, 2)):
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
    return None

# ── Emoji pre-render ───────────────────────────────────────────────────────────
_EMOJI_FONT = ImageFont.load_default(size=72)

def _prerender(emoji):
    canvas = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    with Pilmoji(canvas) as draw:
        draw.text((0, 0), emoji, font=_EMOJI_FONT)
    bbox = canvas.getbbox()
    if bbox:
        canvas = canvas.crop(bbox)
    return np.array(canvas)

_EMOJIS = {item["emoji"]: _prerender(item["emoji"]) for item in ITEMS}

def draw_emoji(frame, emoji, cx, cy):
    img = _EMOJIS.get(emoji)
    if img is None:
        return
    oh, ow = img.shape[:2]
    h, w = frame.shape[:2]
    x, y = cx - ow // 2, cy - oh // 2
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + ow, w), min(y + oh, h)
    if x2 <= x1 or y2 <= y1:
        return
    ex1, ey1 = x1 - x, y1 - y
    ex2, ey2 = ex1 + (x2 - x1), ey1 + (y2 - y1)
    roi = frame[y1:y2, x1:x2]
    src = img[ey1:ey2, ex1:ex2]
    alpha = src[:, :, 3:4] / 255.0
    rgb = cv2.cvtColor(src[:, :, :3], cv2.COLOR_RGB2BGR)
    frame[y1:y2, x1:x2] = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)

# ── Layout ─────────────────────────────────────────────────────────────────────
def make_cards(w, h):
    pad, top = 40, 88
    cw = (w - pad * 3) // 2
    ch = (h - top - pad * 3) // 2
    cards = []
    for i, item in enumerate(ITEMS):
        col, row = i % 2, i // 2
        x1 = pad + col * (cw + pad)
        y1 = top + pad + row * (ch + pad)
        cards.append({**item, "x1": x1, "y1": y1, "x2": x1 + cw, "y2": y1 + ch})
    return cards

# ── Drawing ────────────────────────────────────────────────────────────────────
def _brackets(frame, x1, y1, x2, y2, color, size=22, t=2):
    for (px, py), (dx, dy) in zip(
        [(x1, y1), (x2, y1), (x1, y2), (x2, y2)],
        [(1, 1),  (-1, 1),  (1, -1),  (-1, -1)],
    ):
        cv2.line(frame, (px, py), (px + dx * size, py), color, t)
        cv2.line(frame, (px, py), (px, py + dy * size), color, t)

def draw_card(frame, card, progress, hovered):
    x1, y1, x2, y2 = card["x1"], card["y1"], card["x2"], card["y2"]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    color = card["color"]

    overlay = frame.copy()
    bg = color if hovered else (20, 20, 35)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg, -1)
    cv2.addWeighted(overlay, 0.45 if hovered else 0.3, frame, 0.55 if hovered else 0.7, 0, frame)

    bracket_color = color if hovered else (70, 70, 110)
    _brackets(frame, x1, y1, x2, y2, bracket_color)

    # Emoji centred in upper 60% of card
    draw_emoji(frame, card["emoji"], cx, y1 + int((y2 - y1) * 0.45))

    # Label
    label = card["label"]
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    lbl_color = (255, 255, 255) if hovered else (180, 180, 200)
    cv2.putText(frame, label, (cx - tw // 2, y2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, lbl_color, 2)

    # Progress arc + countdown
    if hovered and progress > 0:
        arc_center = (cx, y1 + 16)
        cv2.ellipse(frame, arc_center, (13, 13), -90, 0, int(360 * progress),
                    (0, 255, 180), 3)
        secs = f"{HOVER_SECONDS * (1 - progress):.1f}s"
        cv2.putText(frame, secs, (cx - 14, y1 + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 180), 1)

# ── Finger detection ───────────────────────────────────────────────────────────
def get_finger(frame, ts):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
    )
    if result.hand_landmarks:
        tip = result.hand_landmarks[0][8]
        h, w = frame.shape[:2]
        return int(tip.x * w), int(tip.y * h)
    return None

def draw_cursor(frame, pos):
    x, y = pos
    cv2.circle(frame, (x, y), 10, (0, 255, 200), -1)
    cv2.circle(frame, (x, y), 13, (255, 255, 255), 2)
    for dx, dy in [(-18, 0), (18, 0), (0, -18), (0, 18)]:
        ox, oy = (x + dx, y + dy)
        mx, my = x + dx // 3, y + dy // 3
        cv2.line(frame, (mx, my), (ox, oy), (0, 255, 200), 2)

# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    # Loading screen while heavy models import (runs once)
    loading = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(loading, "Carregando modelos...", (110, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
    cv2.putText(loading, "Aguarde um momento", (145, 275),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 180, 255), 1)
    cv2.imshow("Vision Lab", loading)
    cv2.waitKey(1)

    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    import fruits, hands, pinch, handgame
    _MODULES = {
        "fruits.py":   fruits,
        "hands.py":    hands,
        "pinch.py":    pinch,
        "handgame.py": handgame,
    }

    cap = open_camera()
    if cap is None:
        print("Erro: nenhuma camera encontrada nos indices 0, 1, 2")
        return

    ts = 0
    hovered_idx = None
    hover_start = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        ts += 1
        now = time.time()

        # Title bar
        cv2.rectangle(frame, (0, 0), (w, 76), (8, 8, 20), -1)
        title = "VISION LAB"
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        cv2.putText(frame, title, (w // 2 - tw // 2, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 255), 2)
        hint = "Aponte o dedo — segure 2s para abrir"
        (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(frame, hint, (w // 2 - hw // 2, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 180, 255), 1)

        cards = make_cards(w, h)
        finger = get_finger(frame, ts)

        new_hovered = None
        if finger:
            fx, fy = finger
            for i, c in enumerate(cards):
                if c["x1"] < fx < c["x2"] and c["y1"] < fy < c["y2"]:
                    new_hovered = i
                    break

        if new_hovered != hovered_idx:
            hovered_idx = new_hovered
            hover_start = now if new_hovered is not None else None

        progress = 0.0
        if hovered_idx is not None and hover_start is not None:
            elapsed = now - hover_start
            progress = min(elapsed / HOVER_SECONDS, 1.0)
            if elapsed >= HOVER_SECONDS:
                selected = cards[hovered_idx]
                cv2.destroyAllWindows()
                cap.release()
                try:
                    _MODULES[selected["file"]].main()
                except SystemExit:
                    pass
                cap = open_camera()
                if cap is None:
                    break
                hovered_idx = None
                hover_start = None
                continue

        for i, card in enumerate(cards):
            draw_card(frame, card, progress if i == hovered_idx else 0.0, i == hovered_idx)

        if finger:
            draw_cursor(frame, finger)

        cv2.imshow("Vision Lab", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


run()
