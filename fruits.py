import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

# =============================
# CARREGAR MODELO
# =============================

model = MobileNetV2(weights="imagenet")

# =============================
# CAPTURA DE VÍDEO
# =============================

cap = cv2.VideoCapture(3)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =============================
    # PRÉ-PROCESSAMENTO
    # =============================

    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # =============================
    # INFERÊNCIA
    # =============================

    preds = model.predict(img, verbose=0)
    decoded = decode_predictions(preds, top=1)[0][0]

    label = decoded[1]
    confidence = decoded[2] * 100

    text = f"{label}: {confidence:.2f}%"

    # =============================
    # EXIBIÇÃO
    # =============================

    cv2.putText(frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2)

    cv2.imshow("Classificador Ao Vivo", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()