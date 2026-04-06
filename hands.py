import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# MODELO HAND LANDMARK
# =========================

model_path = "hand_landmarker.task"  # precisa baixar modelo

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect_for_video(mp_image, timestamp)
    timestamp += 1

    h, w, _ = frame.shape

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            index_tip = hand_landmarks[8]

            cx = int(index_tip.x * w)
            cy = int(index_tip.y * h)

            cv2.circle(frame, (cx, cy), 30, (0, 255, 0), -1)

    cv2.imshow("Hands Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
