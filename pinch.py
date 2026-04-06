import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

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

cap = cv2.VideoCapture(3)

timestamp = 0

# Objeto controlável
obj_x, obj_y = 300, 300
obj_radius = 40
dragging = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect_for_video(mp_image, timestamp)
    timestamp += 1

    h, w, _ = frame.shape

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        index_tip = hand[8]
        thumb_tip = hand[4]

        ix, iy = int(index_tip.x * w), int(index_tip.y * h)
        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

        # Desenhar dedos
        cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)
        cv2.circle(frame, (tx, ty), 10, (255, 0, 0), -1)

        # Distância entre dedos
        dist = math.hypot(ix - tx, iy - ty)

        # Threshold ajustável
        if dist < 40:
            dragging = True
        else:
            dragging = False

        # Se estiver "pinçando", move objeto
        if dragging:
            obj_x, obj_y = ix, iy

    # Desenhar objeto
    cv2.circle(frame, (obj_x, obj_y), obj_radius, (0, 0, 255), -1)

    cv2.imshow("Controle por Gestos", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
