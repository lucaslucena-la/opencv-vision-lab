import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import random
import math

# =========================
# CONFIG MEDIAPIPE
# =========================

model_path = "hand_landmarker.task"

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# =========================
# JOGO
# =========================

cap = cv2.VideoCapture(3)
timestamp = 0

score = 0

player_radius = 25
coin_radius = 20

coin_x = random.randint(100, 500)
coin_y = random.randint(100, 400)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect_for_video(mp_image, timestamp)
    timestamp += 1

    player_x, player_y = w // 2, h // 2

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        index_tip = hand[8]

        player_x = int(index_tip.x * w)
        player_y = int(index_tip.y * h)

    # =========================
    # COLISÃO
    # =========================

    distance = math.hypot(player_x - coin_x, player_y - coin_y)

    if distance < player_radius + coin_radius:
        score += 1
        coin_x = random.randint(50, w - 50)
        coin_y = random.randint(50, h - 50)

    # =========================
    # DESENHO
    # =========================

    # Moeda
    cv2.circle(frame, (coin_x, coin_y), coin_radius, (0, 255, 255), -1)

    # Jogador
    cv2.circle(frame, (player_x, player_y), player_radius, (255, 0, 0), -1)

    # Pontuação
    cv2.putText(frame,
                f"Pontos: {score}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Mini Jogo - Pegue as Moedas", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
