import cv2
import math
import mediapipe as mp
import numpy as np
import sys
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageFont
from pilmoji import Pilmoji

model_path = "hand_landmarker.task"

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)


def prerender_emoji(emoji, size):
    font = ImageFont.load_default(size=size)
    canvas = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
    with Pilmoji(canvas) as draw:
        draw.text((0, 0), emoji, font=font)
    bbox = canvas.getbbox()
    if bbox:
        canvas = canvas.crop(bbox)
    return np.array(canvas)

COOKIE_BASE = prerender_emoji("🍪", 512)

def draw_cookie(frame, x, y, radius):
    size = max(radius * 2, 1)
    resized = cv2.resize(COOKIE_BASE, (size, size), interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]
    h, w = frame.shape[:2]
    x0, y0 = x - rw // 2, y - rh // 2
    fx1, fy1 = max(x0, 0), max(y0, 0)
    fx2, fy2 = min(x0 + rw, w), min(y0 + rh, h)
    if fx2 <= fx1 or fy2 <= fy1:
        return frame
    ex1, ey1 = fx1 - x0, fy1 - y0
    ex2, ey2 = ex1 + (fx2 - fx1), ey1 + (fy2 - fy1)
    roi = frame[fy1:fy2, fx1:fx2]
    src = resized[ey1:ey2, ex1:ex2]
    alpha = src[:, :, 3:4] / 255.0
    rgb = cv2.cvtColor(src[:, :, :3], cv2.COLOR_RGB2BGR)
    frame[fy1:fy2, fx1:fx2] = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
    return frame

def pinch_center(hand, w, h):
    ix, iy = int(hand[8].x * w), int(hand[8].y * h)
    tx, ty = int(hand[4].x * w), int(hand[4].y * h)
    dist = math.hypot(ix - tx, iy - ty)
    cx, cy = (ix + tx) // 2, (iy + ty) // 2
    return dist, cx, cy

def main():
    cap = cv2.VideoCapture(0)
    ts = int(time.time() * 1000)
    x_hover_start = None
    obj_x, obj_y = 300, 300
    obj_radius = 40
    pinch_dist_ref = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect_for_video(mp_image, ts)
        ts += 1

        h, w, _ = frame.shape
        hands = result.hand_landmarks

        fx, fy = None, None
        if hands:
            tip = hands[0][8]
            fx, fy = int(tip.x * w), int(tip.y * h)

        if len(hands) == 1:
            pinch_dist_ref = None
            dist, cx, cy = pinch_center(hands[0], w, h)
            if dist < 40:
                obj_x, obj_y = cx, cy

        elif len(hands) == 2:
            dist0, cx0, cy0 = pinch_center(hands[0], w, h)
            dist1, cx1, cy1 = pinch_center(hands[1], w, h)
            if dist0 < 40 and dist1 < 40:
                spread = math.hypot(cx0 - cx1, cy0 - cy1)
                if pinch_dist_ref is None:
                    pinch_dist_ref = (spread, obj_radius)
                else:
                    ref_spread, ref_radius = pinch_dist_ref
                    obj_radius = max(10, min(300, int(ref_radius * spread / ref_spread)))
            else:
                pinch_dist_ref = None

        frame = draw_cookie(frame, obj_x, obj_y, obj_radius)

        _bx1, _bx2 = w - 62, w - 10
        _now = time.time()
        _in_x = fx is not None and _bx1 < fx < _bx2 and 10 < fy < 62
        if _in_x:
            if x_hover_start is None:
                x_hover_start = _now
            _prog = min((_now - x_hover_start) / 1.5, 1.0)
            cv2.rectangle(frame, (_bx1, 10), (_bx2, 62), (0, 0, 120), -1)
            cv2.rectangle(frame, (_bx1, int(62 - 52 * _prog)), (_bx2, 62), (40, 40, 255), -1)
            if _prog >= 1.0:
                cap.release()
                cv2.destroyAllWindows()
                return
        else:
            x_hover_start = None
            cv2.rectangle(frame, (_bx1, 10), (_bx2, 62), (0, 0, 80), -1)
        cv2.rectangle(frame, (_bx1, 10), (_bx2, 62), (100, 100, 210), 2)
        cv2.putText(frame, "X", (_bx1 + 14, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow("Controle por Gestos", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
