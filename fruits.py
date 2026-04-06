import cv2
import numpy as np
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
cap = cv2.VideoCapture(3)

frame_count = 0

# Texto inicial
display_texts = ["Procurando fruta..."]

# Armazena fruta principal detectada
main_label = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Processa somente a cada N frames
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        results = classify_fruit(frame)

        if results:
            # Primeira fruta detectada
            main_label = results[0][0]

            # Exibe até 3 resultados
            display_texts = [
                f"{label}: {conf:.2f}%"
                for label, conf in results[:3]
            ]
        else:
            display_texts = ["Nenhuma fruta detectada"]
            main_label = None

    # DESENHA EMOJI
    if main_label:
        frame = draw_emoji(frame, main_label)

    # Mostra textos na tela
    y = 30
    for text in display_texts:
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        y += 30

    # Exibe janela
    cv2.imshow("Classificador de Frutas", frame)

    # ESC fecha programa
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Libera câmera
cap.release()

# Fecha janelas
cv2.destroyAllWindows()