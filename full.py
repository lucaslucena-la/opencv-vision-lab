import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =============================
# CORES NEON POR EMOÇÃO (BGR)
# =============================
EMOTION_STYLE = {
    "Raiva":    {"color": (0,   0,   255), "emoji": ">:("},
    "Nojo":     {"color": (0,   128, 0  ), "emoji": ":/"},
    "Medo":     {"color": (128, 0,   128), "emoji": "D:"},
    "Feliz":    {"color": (0,   255, 255), "emoji": ":D"},
    "Triste":   {"color": (255, 100, 0  ), "emoji": ":("},
    "Surpreso": {"color": (0,   165, 255), "emoji": ":O"},
    "Neutro":   {"color": (200, 200, 200), "emoji": ":|"},
}

# Cor padrão caso emoção não encontrada
DEFAULT_COLOR = (0, 255, 0)

# =============================
# UTILITÁRIOS DE DESENHO
# =============================

def draw_rounded_rect(img, x, y, w, h, color, radius=12, thickness=2):
    """Desenha retângulo com cantos arredondados estilo sci-fi."""
    x1, y1, x2, y2 = x, y, x + w, y + h
    # Linhas
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    # Arcos nos cantos
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90,  color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90,  color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius),  90, 0, 90,  color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius),   0, 0, 90,  color, thickness)


def draw_corner_brackets(img, x, y, w, h, color, size=18, thickness=2):
    """Desenha colchetes nos cantos do rosto (estilo scanner)."""
    x2, y2 = x + w, y + h
    pts = [
        ((x, y),  [(x, y + size), (x + size, y)]),
        ((x2, y), [(x2, y + size), (x2 - size, y)]),
        ((x, y2), [(x, y2 - size), (x + size, y2)]),
        ((x2,y2), [(x2, y2 - size), (x2 - size, y2)]),
    ]
    for origin, ends in pts:
        for end in ends:
            cv2.line(img, origin, end, color, thickness)


def draw_semi_transparent_box(img, x, y, w, h, color=(0, 0, 0), alpha=0.5):
    """Caixa semitransparente para texto."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_emotion_bars(img, probs, labels, styles, x, y, bar_w=130, bar_h=12, spacing=20):
    """Barras de probabilidade coloridas para cada emoção."""
    panel_h = len(labels) * spacing + 10
    draw_semi_transparent_box(img, x - 5, y - 5, bar_w + 90, panel_h, alpha=0.55)

    for i, (label, prob) in enumerate(zip(labels, probs)):
        bar_y = y + i * spacing
        color = styles.get(label, {}).get("color", DEFAULT_COLOR)
        filled = int(prob * bar_w)

        # Fundo da barra
        cv2.rectangle(img, (x, bar_y), (x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        # Barra preenchida
        cv2.rectangle(img, (x, bar_y), (x + filled, bar_y + bar_h), color, -1)
        # Borda da barra
        cv2.rectangle(img, (x, bar_y), (x + bar_w, bar_y + bar_h), color, 1)

        text = f"{label}: {prob*100:.0f}%"
        cv2.putText(img, text,
                    (x + bar_w + 5, bar_y + bar_h - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    color, 1, cv2.LINE_AA)


def draw_hud_panel(img, emotion, gender, age, emoji, x, y, color):
    """Painel HUD abaixo do rosto com emoção, gênero e idade."""
    panel_w = 220
    panel_h = 56
    px, py = x, y + 6

    draw_semi_transparent_box(img, px, py, panel_w, panel_h, alpha=0.6)

    # Linha decorativa superior
    cv2.line(img, (px, py), (px + panel_w, py), color, 1)

    # Emoji + emoção
    cv2.putText(img, f"{emoji}  {emotion}",
                (px + 6, py + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2, cv2.LINE_AA)

    # Gênero e idade
    cv2.putText(img, f"{gender}  |  {age}",
                (px + 6, py + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (220, 220, 220), 1, cv2.LINE_AA)

    # Linha decorativa inferior
    cv2.line(img, (px, py + panel_h), (px + panel_w, py + panel_h), color, 1)


def draw_scanline_overlay(img, alpha=0.06):
    """Efeito de scanlines sutis para aparência sci-fi."""
    overlay = img.copy()
    h, w = img.shape[:2]
    for line_y in range(0, h, 4):
        cv2.line(overlay, (0, line_y), (w, line_y), (0, 0, 0), 1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_top_hud(img, mode, face_count, fps):
    """Barra superior com informações gerais."""
    h, w = img.shape[:2]
    draw_semi_transparent_box(img, 0, 0, w, 44, alpha=0.65)
    cv2.line(img, (0, 44), (w, 44), (0, 220, 220), 1)

    mode_text = {0: "TODOS", 1: "EMOCAO", 2: "GENERO", 3: "IDADE"}
    cv2.putText(img, f"MODO: {mode_text.get(mode, '?')}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 220), 2, cv2.LINE_AA)

    cv2.putText(img, f"ROSTOS: {face_count}",
                (w // 2 - 50, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 220), 2, cv2.LINE_AA)

    cv2.putText(img, f"FPS: {fps}",
                (w - 100, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 220), 2, cv2.LINE_AA)


def draw_key_hints(img):
    """Dicas de teclas no canto inferior."""
    h, w = img.shape[:2]
    hints = "[0] Todos  [1] Emocao  [2] Genero  [3] Idade  [ESC] Sair"
    draw_semi_transparent_box(img, 0, h - 28, w, 28, alpha=0.6)
    cv2.putText(img, hints, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)


# =============================
# CÂMERA
# =============================

def open_camera(indices=(0, 1, 2)):
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap, idx
            cap.release()
    return None, None


# =============================
# MODELOS
# =============================

emotion_model = load_model("fer2013_mini_XCEPTION.102-0.66.hdf5", compile=False)
emotion_labels = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]

age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")
gender_list = ['Masculino', 'Feminino']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


# =============================
# PREDIÇÕES
# =============================

def predict_emotion(face_gray):
    h, w = emotion_model.input_shape[1:3]
    face = cv2.resize(face_gray, (w, h)).astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    preds = emotion_model.predict(face, verbose=0)
    return emotion_labels[np.argmax(preds)], preds[0]

def predict_age(face_color):
    blob = cv2.dnn.blobFromImage(face_color, 1.0, (227, 227),
        (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    age_net.setInput(blob)
    return age_list[age_net.forward()[0].argmax()]

def predict_gender(face_color):
    blob = cv2.dnn.blobFromImage(face_color, 1.0, (227, 227),
        (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    gender_net.setInput(blob)
    return gender_list[gender_net.forward()[0].argmax()]


# =============================
# LOOP PRINCIPAL
# =============================

cap, cam_idx = open_camera()

if cap is None:
    print("Erro ao acessar a camera. Indices testados: 0, 1, 2")
    raise SystemExit(1)

print(f"Camera ativa no indice {cam_idx}")

mode = 0
import time
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS
    now = time.time()
    fps = int(1.0 / max(now - prev_time, 1e-5))
    prev_time = now

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Painel de barras de emoção — só no modo 0 ou 1, para o primeiro rosto
    bars_drawn = False

    for i, (x, y, w, h) in enumerate(faces):
        face_gray  = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]

        emotion, probs = predict_emotion(face_gray)
        color   = EMOTION_STYLE.get(emotion, {}).get("color", DEFAULT_COLOR)
        emoji   = EMOTION_STYLE.get(emotion, {}).get("emoji", "")

        gender = predict_gender(face_color) if mode in (0, 2) else "—"
        age    = predict_age(face_color)    if mode in (0, 3) else "—"

        # Retângulo com cantos arredondados
        draw_rounded_rect(frame, x, y, w, h, color, radius=14, thickness=2)
        draw_corner_brackets(frame, x, y, w, h, color, size=20, thickness=2)

        # HUD abaixo do rosto
        if mode == 0:
            draw_hud_panel(frame, emotion, gender, age, emoji, x, y + h, color)
        elif mode == 1:
            draw_hud_panel(frame, emotion, "—", "—", emoji, x, y + h, color)
        elif mode == 2:
            draw_hud_panel(frame, "—", gender, "—", "", x, y + h, color)
        elif mode == 3:
            draw_hud_panel(frame, "—", "—", age, "", x, y + h, color)

        # Barras de probabilidade — apenas para o primeiro rosto, no canto esquerdo
        if not bars_drawn and mode in (0, 1):
            draw_emotion_bars(frame, probs, emotion_labels, EMOTION_STYLE,
                              x=8, y=55, bar_w=110)
            bars_drawn = True

    # Scanlines sutis
    draw_scanline_overlay(frame, alpha=0.07)

    # HUD superior e dicas
    draw_top_hud(frame, mode, len(faces), fps)
    draw_key_hints(frame)

    cv2.imshow("Face AI - Sci-Fi HUD", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('0'): mode = 0
    elif key == ord('1'): mode = 1
    elif key == ord('2'): mode = 2
    elif key == ord('3'): mode = 3

cap.release()
cv2.destroyAllWindows()