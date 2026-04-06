import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =============================
# CONFIGURAÇÕES
# =============================

emotion_model = load_model("fer2013_mini_XCEPTION.102-0.66.hdf5", compile=False)
emotion_labels = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]

age_net = cv2.dnn.readNetFromCaffe(
    "age_deploy.prototxt",
    "age_net.caffemodel"
)

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

gender_net = cv2.dnn.readNetFromCaffe(
    "gender_deploy.prototxt",
    "gender_net.caffemodel"
)

gender_list = ['Masculino', 'Feminino']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# =============================
# FUNÇÕES
# =============================

def predict_emotion(face_gray):
    h, w = emotion_model.input_shape[1:3]
    face = cv2.resize(face_gray, (w, h))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    preds = emotion_model.predict(face, verbose=0)
    return emotion_labels[np.argmax(preds)]

def predict_age(face_color):
    blob = cv2.dnn.blobFromImage(
        face_color, 1.0, (227, 227),
        (78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False
    )
    age_net.setInput(blob)
    age_preds = age_net.forward()
    return age_list[age_preds[0].argmax()]

def predict_gender(face_color):
    blob = cv2.dnn.blobFromImage(
        face_color, 1.0, (227, 227),
        (78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False
    )
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    return gender_list[gender_preds[0].argmax()]

# =============================
# LOOP PRINCIPAL
# =============================

cap = cv2.VideoCapture(3)

mode = 0  # 0 = todos | 1 = emoção | 2 = gênero | 3 = idade

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

        face_gray = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]

        label = ""

        if mode == 1:
            emotion = predict_emotion(face_gray)
            label = f"{emotion}"

        elif mode == 2:
            gender = predict_gender(face_color)
            label = f"{gender}"

        elif mode == 3:
            age = predict_age(face_color)
            label = f"{age}"

        elif mode == 0:
            emotion = predict_emotion(face_gray)
            gender = predict_gender(face_color)
            age = predict_age(face_color)
            label = f"{emotion} | {gender} | {age}"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame, label,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

    # Exibe modo atual
    mode_text = {
        0: "Modo: TODOS",
        1: "Modo: EMOCAO",
        2: "Modo: GENERO",
        3: "Modo: IDADE"
    }

    cv2.putText(frame, mode_text[mode],
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2)

    cv2.imshow("win", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('0'):
        mode = 0
    elif key == ord('1'):
        mode = 1
    elif key == ord('2'):
        mode = 2
    elif key == ord('3'):
        mode = 3

cap.release()
cv2.destroyAllWindows()