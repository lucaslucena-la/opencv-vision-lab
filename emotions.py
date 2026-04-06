import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Carregar modelo de emoção
emotion_model = load_model("fer2013_mini_XCEPTION.102-0.66.hdf5", compile=False)

# Labels padrão do FER2013
emotion_labels = [
    "Raiva",
    "Nojo",
    "Medo",
    "Feliz",
    "Triste",
    "Surpreso",
    "Neutro"
]

# Detector facial
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Webcam
cap = cv2.VideoCapture(3)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        # Recorte do rosto
        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (64, 64))  # tamanho correto

        face = face.astype("float32") / 255.0

        # adiciona canal (grayscale → 1 canal)
        face = np.expand_dims(face, axis=-1)

        # adiciona batch dimension
        face = np.expand_dims(face, axis=0)

        # Predição
        preds = emotion_model.predict(face, verbose=0)
        emotion = emotion_labels[np.argmax(preds)]

        # Desenho na tela
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2)
        for i, (emotion_name, prob) in enumerate(zip(emotion_labels, preds[0])):
            text = f"{emotion_name}: {prob*100:.1f}%"
            cv2.putText(frame, text,
                (10, 30 + i*25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1)

    cv2.imshow("Deteccao de Emocao - Mostra CC", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
