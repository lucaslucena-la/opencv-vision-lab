import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageFont
from pilmoji import Pilmoji

model_path = "gesture_recognizer.task"

if not os.path.exists(model_path):
    print("Downloading gesture_recognizer.task...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
        model_path,
    )
    print("Download completo.")

BaseOptions = python.BaseOptions
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
VisionRunningMode = vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
)

recognizer = GestureRecognizer.create_from_options(options)

GESTURE_EMOJI = {
    "Thumb_Up": "👍",
    "Thumb_Down": "👎",
    "Open_Palm": "🖐️",
    "Victory": "✌️",
    "Closed_Fist": "✊",
    "Pointing_Up": "☝️",
    "ILoveYou": "🤟",
}


def finger_extended(lm, tip, pip):
    return lm[tip].y < lm[pip].y


def finger_curled(lm, tip, pip):
    return lm[tip].y > lm[pip].y


def detect_custom_gesture(lm):
    if (
        finger_extended(lm, 8, 6)
        and finger_extended(lm, 20, 18)
        and finger_curled(lm, 12, 10)
        and finger_curled(lm, 16, 14)
    ):
        return "🤘"
    if (
        finger_extended(lm, 12, 10)
        and finger_curled(lm, 8, 6)
        and finger_curled(lm, 16, 14)
        and finger_curled(lm, 20, 18)
    ):
        return "🖕"
    if (
        lm[4].y < lm[3].y
        and finger_extended(lm, 20, 18)
        and finger_curled(lm, 8, 6)
        and finger_curled(lm, 12, 10)
        and finger_curled(lm, 16, 14)
    ):
        return "🤙"
    return None


EMOJI_FONT = ImageFont.load_default(size=120)


def prerender_emoji(emoji):
    canvas = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    with Pilmoji(canvas) as draw:
        draw.text((0, 0), emoji, font=EMOJI_FONT)
    bbox = canvas.getbbox()
    if bbox:
        canvas = canvas.crop(bbox)
    return np.array(canvas)


ALL_EMOJIS = list(GESTURE_EMOJI.values()) + ["🤘", "🖕", "🤙"]
RENDERED = {e: prerender_emoji(e) for e in ALL_EMOJIS}


def draw_emoji(frame, emoji):
    overlay = RENDERED.get(emoji)
    if overlay is None:
        return frame
    oh, ow = overlay.shape[:2]
    h, w = frame.shape[:2]
    x = w // 2 - ow // 2
    y = h - oh - 20
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + ow, w), min(y + oh, h)
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
    roi = frame[y1:y2, x1:x2]
    src = overlay[oy1:oy2, ox1:ox2]
    alpha = src[:, :, 3:4] / 255.0
    rgb = cv2.cvtColor(src[:, :, :3], cv2.COLOR_RGB2BGR)
    frame[y1:y2, x1:x2] = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
    return frame


cap = cv2.VideoCapture(0)

timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = recognizer.recognize_for_video(mp_image, timestamp)
    timestamp += 1

    h, w, _ = frame.shape

    emoji = None
    if result.gestures:
        gesture_name = result.gestures[0][0].category_name
        emoji = GESTURE_EMOJI.get(gesture_name)

    if emoji is None and result.hand_landmarks:
        emoji = detect_custom_gesture(result.hand_landmarks[0])

    if emoji:
        frame = draw_emoji(frame, emoji)

    cv2.imshow("Hands Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
