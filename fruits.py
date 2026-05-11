import cv2
import numpy as np
import sys
import time
import mediapipe as mp
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision
from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0,
    preprocess_input,
    decode_predictions
)
from PIL import Image, ImageFont
from pilmoji import Pilmoji

# =============================
# CONFIGURAÇÕES
# =============================

# Confiança mínima para considerar uma fruta detectada
CONFIDENCE_THRESHOLD = 40

# Processa a classificação a cada 3 frames
# para melhorar performance
PROCESS_EVERY_N_FRAMES = 3

# Lista de palavras-chave consideradas frutas
FRUIT_KEYWORDS = {
    "banana",
    "pineapple",
    "lemon",
    "orange",
    "pomegranate",
    "fig",
    "jackfruit",
    "custard_apple",
    "apple",
    "grape",
    "pear",
    "peach",
    "strawberry",
    "coconut"
}

# Dicionário que relaciona cada fruta com um emoji
FRUIT_EMOJIS = {
    "banana": "🍌",
    "pineapple": "🍍",
    "lemon": "🍋",
    "orange": "🍊",
    "pomegranate": "🍎",
    "fig": "🍈",
    "jackfruit": "🍈",
    "custard_apple": "🍏",
    "apple": "🍎",
    "grape": "🍇",
    "pear": "🍐",
    "peach": "🍑",
    "strawberry": "🍓",
    "coconut": "🥥"
}

# =============================
# MODELO
# =============================

# Carrega modelo pré-treinado EfficientNetB0
# com pesos do ImageNet
model = EfficientNetB0(weights="imagenet")

# Hand landmarker para botão X
_hand_landmarker = _mp_vision.HandLandmarker.create_from_options(
    _mp_vision.HandLandmarkerOptions(
        base_options=_mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=_mp_vision.RunningMode.VIDEO,
        num_hands=1,
    )
)

# =============================
# EMOJI
# =============================

# Fonte usada para renderizar emojis
EMOJI_FONT = ImageFont.load_default(size=100)


def prerender_emoji(emoji):
    """
    Renderiza o emoji uma única vez em memória
    para evitar travamentos durante execução.
    """
    canvas = Image.new("RGBA", (200, 200), (0, 0, 0, 0))

    with Pilmoji(canvas) as draw:
        draw.text((0, 0), emoji, font=EMOJI_FONT)

    bbox = canvas.getbbox()

    # Remove espaço vazio ao redor do emoji
    if bbox:
        canvas = canvas.crop(bbox)

    return np.array(canvas)


# PRÉ-RENDERIZA TODOS
# Cria todos os emojis antes do loop principal
RENDERED = {
    fruit: prerender_emoji(emoji)
    for fruit, emoji in FRUIT_EMOJIS.items()
}


def draw_emoji(frame, label):
    """
    Desenha o emoji correspondente à fruta detectada
    no canto superior direito da tela.
    """
    label = label.lower()

    selected = None

    # Procura qual fruta foi identificada
    for fruit in FRUIT_EMOJIS:
        if fruit in label:
            selected = fruit
            break

    # Se não encontrou fruta, retorna frame original
    if selected is None:
        return frame

    overlay = RENDERED[selected]

    # Altura e largura do emoji
    oh, ow = overlay.shape[:2]

    # Posição do emoji no canto superior direito
    x = frame.shape[1] - ow - 20
    y = 20

    # Região onde o emoji será desenhado
    roi = frame[y:y+oh, x:x+ow]

    # Transparência do emoji
    alpha = overlay[:, :, 3:4] / 255.0

    # Converte RGB para BGR (OpenCV)
    rgb = cv2.cvtColor(overlay[:, :, :3], cv2.COLOR_RGB2BGR)

    # Mistura emoji com frame
    frame[y:y+oh, x:x+ow] = (
        rgb * alpha + roi * (1 - alpha)
    ).astype(np.uint8)

    return frame


# =============================
# FUNÇÕES
# =============================

def preprocess_frame(frame):
    """
    Faz o pré-processamento da imagem
    para entrada no modelo.
    """
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


def is_fruit(label):
    """
    Verifica se o rótulo previsto
    pertence a uma fruta.
    """
    label = label.lower()
    return any(fruit in label for fruit in FRUIT_KEYWORDS)


def classify_fruit(frame):
    """
    Realiza a classificação da imagem
    e retorna as frutas detectadas.
    """
    img = preprocess_frame(frame)

    preds = model.predict(img, verbose=0)

    # Retorna as 5 previsões mais prováveis
    decoded = decode_predictions(preds, top=5)[0]

    results = []

    for item in decoded:
        label = item[1]
        confidence = item[2] * 100

        # Filtra apenas frutas
        if is_fruit(label) and confidence >= CONFIDENCE_THRESHOLD:
            results.append((label, confidence))

    return results


# =============================
# CÂMERA
# =============================

# Abre câmera
def _open_camera(indices=(0, 1, 2)):
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
    return None

def main():
    cap = _open_camera()
    if cap is None:
        print("Erro: nenhuma camera encontrada nos indices 0, 1, 2")
        return

    frame_count = 0
    ts = int(time.time() * 1000)
    x_hover_start = None
    display_texts = ["Procurando fruta..."]
    main_label = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1
        ts += 1

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            results = classify_fruit(frame)
            if results:
                main_label = results[0][0]
                display_texts = [
                    f"{label}: {conf:.2f}%"
                    for label, conf in results[:3]
                ]
            else:
                display_texts = ["Nenhuma fruta detectada"]
                main_label = None

        if main_label:
            frame = draw_emoji(frame, main_label)

        y = 30
        for text in display_texts:
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y += 30

        _lm_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _lm_res = _hand_landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=_lm_rgb), ts
        )
        fx, fy = None, None
        if _lm_res.hand_landmarks:
            _tip = _lm_res.hand_landmarks[0][8]
            _fh, _fw = frame.shape[:2]
            fx, fy = int(_tip.x * _fw), int(_tip.y * _fh)

        _ffw = frame.shape[1]
        _bx1, _bx2 = _ffw - 62, _ffw - 10
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

        cv2.imshow("Classificador de Frutas", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()