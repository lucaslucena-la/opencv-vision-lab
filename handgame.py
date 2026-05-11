import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import random
import math
import time
import os
import sys

# =========================
# CONFIG MEDIAPIPE
# =========================
# Caminho do modelo de detecção da mão
model_path = "hand_landmarker.task"

# Importação dos componentes do MediaPipe
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Configuração do detector para vídeo em tempo real
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

# Criação do detector de mãos
landmarker = HandLandmarker.create_from_options(options)

# =========================
# CONFIGURAÇÕES GERAIS DO JOGO
# =========================
RANKING_FILE = "ranking.txt"   # arquivo onde o ranking é salvo
GAME_TIME = 10                 # tempo total da partida
PLAYER_RADIUS = 20             # tamanho do cursor do jogador

# Coordenadas do botão "JOGAR"
BUTTON_X1, BUTTON_Y1 = 200, 325
BUTTON_X2, BUTTON_Y2 = 420, 405

# Caixas das letras das iniciais
LETTER_BOXES = [
    (180, 80, 240, 140),
    (300, 80, 360, 140),
    (420, 80, 480, 140)
]

# Lista de letras do alfabeto
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# =========================
# FUNÇÃO PARA CARREGAR RANKING
# =========================
def carregar_ranking():
    ranking = {}

    # Verifica se o arquivo existe
    if os.path.exists(RANKING_FILE):
        with open(RANKING_FILE, "r") as file:
            for line in file:
                line = line.strip()

                # Ignora linhas vazias
                if not line:
                    continue

                parts = line.split(",")

                # Separa nome e pontuação
                if len(parts) == 2:
                    name = parts[0].strip()

                    try:
                        score = int(parts[1].strip())
                        ranking[name] = score
                    except ValueError:
                        continue

    return ranking


# =========================
# FUNÇÃO PARA SALVAR PONTUAÇÃO
# =========================
def salvar_pontuacao(iniciais, score):
    ranking = carregar_ranking()

    # Atualiza apenas se for maior pontuação
    if iniciais not in ranking or score > ranking[iniciais]:
        ranking[iniciais] = score

    # Reescreve o arquivo com os dados atualizados
    with open(RANKING_FILE, "w") as file:
        for name, points in ranking.items():
            file.write(f"{name},{points}\n")


# =========================
# FUNÇÃO DE DETECÇÃO DO DEDO
# =========================
def detectar_dedo(frame, timestamp):
    # Converte para RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Cria imagem do MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Detecta mão
    result = landmarker.detect_for_video(mp_image, timestamp)

    h, w, _ = frame.shape

    # Posição padrão no centro
    x, y = w // 2, h // 2

    # Caso a mão seja detectada
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # Ponta do dedo indicador
        index_tip = hand[8]

        x = int(index_tip.x * w)
        y = int(index_tip.y * h)

    return x, y


# =========================
# TELA DE SELEÇÃO DE INICIAIS
# =========================
def selecionar_iniciais_por_gesto(cap, timestamp_ref):
    iniciais = ["A", "A", "A"]
    selected = 0
    hover_time = None
    x_hover_start_x = None

    while True:
        ret, frame = cap.read()
        if not ret:
            return "AAA", timestamp_ref

        frame = cv2.flip(frame, 1)
        timestamp_ref += 1

        # Detecta posição do dedo
        x, y = detectar_dedo(frame, timestamp_ref)

        # Título da tela
        cv2.putText(frame, "Escolha as iniciais por gesto", (120, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Desenha caixas das letras
        for i, (x1, y1, x2, y2) in enumerate(LETTER_BOXES):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), -1)
            cv2.putText(frame, iniciais[i], (x1 + 18, y1 + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Instruções
        cv2.putText(frame, "Passe o dedo na caixa para trocar", (120, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, "Segure 2s no botao OK", (170, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Botão OK
        cv2.rectangle(frame, (260, 300), (380, 360), (0, 200, 0), -1)
        cv2.putText(frame, "OK", (300, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        # Controle do tempo entre trocas
        if "last_change_time" not in locals():
            last_change_time = 0

        # Troca letras ao passar o dedo
        for i, (x1, y1, x2, y2) in enumerate(LETTER_BOXES):
            if x1 < x < x2 and y1 < y < y2:
                selected = i
                current_time = time.time()

                if current_time - last_change_time > 0.3:
                    idx = LETTERS.index(iniciais[i])
                    iniciais[i] = LETTERS[(idx + 1) % len(LETTERS)]
                    last_change_time = current_time

        # Confirmação no botão OK
        if 260 < x < 380 and 300 < y < 360:
            if hover_time is None:
                hover_time = time.time()

            if time.time() - hover_time >= 2:
                return "".join(iniciais), timestamp_ref
        else:
            hover_time = None

        # Cursor do jogador
        cv2.circle(frame, (x, y), 15, (255, 0, 0), -1)

        _fw_x = frame.shape[1]
        _bx1_x, _bx2_x = _fw_x - 62, _fw_x - 10
        _now_x = time.time()
        if _bx1_x < x < _bx2_x and 10 < y < 62:
            if x_hover_start_x is None:
                x_hover_start_x = _now_x
            _prog_x = min((_now_x - x_hover_start_x) / 1.5, 1.0)
            cv2.rectangle(frame, (_bx1_x, 10), (_bx2_x, 62), (0, 0, 120), -1)
            cv2.rectangle(frame, (_bx1_x, int(62 - 52 * _prog_x)), (_bx2_x, 62), (40, 40, 255), -1)
            if _prog_x >= 1.0:
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)
        else:
            x_hover_start_x = None
            cv2.rectangle(frame, (_bx1_x, 10), (_bx2_x, 62), (0, 0, 80), -1)
        cv2.rectangle(frame, (_bx1_x, 10), (_bx2_x, 62), (100, 100, 210), 2)
        cv2.putText(frame, "X", (_bx1_x + 14, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow("Mini Jogo - Pegue as Moedas", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return None, timestamp_ref


# =========================
# INÍCIO DA CÂMERA
# =========================
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
    timestamp = int(time.time() * 1000)

    while True:
        score = 0
        high_score = 0
        coin_radius = 20
        coin_x = random.randint(100, 500)
        coin_y = random.randint(100, 400)
        game_started = False
        hover_start_time = None
        start_time = None
        x_hover_start_g = None

        iniciais, timestamp = selecionar_iniciais_por_gesto(cap, timestamp)
        if iniciais is None:
            break

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            timestamp += 1
            player_x, player_y = detectar_dedo(frame, timestamp)

            if not game_started:
                cv2.rectangle(frame, (BUTTON_X1, BUTTON_Y1),
                              (BUTTON_X2, BUTTON_Y2), (0, 255, 0), -1)
                cv2.putText(frame, f"Jogador: {iniciais}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "JOGAR", (255, 375),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.circle(frame, (player_x, player_y), 20, (255, 0, 0), -1)

                if BUTTON_X1 < player_x < BUTTON_X2 and BUTTON_Y1 < player_y < BUTTON_Y2:
                    if hover_start_time is None:
                        hover_start_time = time.time()
                    if time.time() - hover_start_time >= 3:
                        game_started = True
                        start_time = time.time()
                else:
                    hover_start_time = None

                _fw_g = frame.shape[1]
                _bgx1, _bgx2 = _fw_g - 62, _fw_g - 10
                _now_g = time.time()
                if _bgx1 < player_x < _bgx2 and 10 < player_y < 62:
                    if x_hover_start_g is None:
                        x_hover_start_g = _now_g
                    _prog_g = min((_now_g - x_hover_start_g) / 1.5, 1.0)
                    cv2.rectangle(frame, (_bgx1, 10), (_bgx2, 62), (0, 0, 120), -1)
                    cv2.rectangle(frame, (_bgx1, int(62 - 52 * _prog_g)), (_bgx2, 62), (40, 40, 255), -1)
                    if _prog_g >= 1.0:
                        cap.release()
                        cv2.destroyAllWindows()
                        sys.exit(0)
                else:
                    x_hover_start_g = None
                    cv2.rectangle(frame, (_bgx1, 10), (_bgx2, 62), (0, 0, 80), -1)
                cv2.rectangle(frame, (_bgx1, 10), (_bgx2, 62), (100, 100, 210), 2)
                cv2.putText(frame, "X", (_bgx1 + 14, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                cv2.imshow("Mini Jogo - Pegue as Moedas", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    game_started = False
                    break
                continue

            remaining_time = GAME_TIME - int(time.time() - start_time)
            if remaining_time <= 0:
                break

            distance = math.hypot(player_x - coin_x, player_y - coin_y)
            if distance < PLAYER_RADIUS + coin_radius:
                score += 1
                high_score = max(high_score, score)
                coin_radius = max(8, 20 - score)
                coin_x = random.randint(50, frame.shape[1] - 50)
                coin_y = random.randint(50, frame.shape[0] - 50)

            cv2.circle(frame, (coin_x, coin_y), coin_radius, (0, 255, 255), -1)
            cv2.circle(frame, (player_x, player_y), PLAYER_RADIUS, (255, 0, 0), -1)
            cv2.putText(frame, f"Pontos: {score}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Tempo: {remaining_time}s", (430, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            _fw_g = frame.shape[1]
            _bgx1, _bgx2 = _fw_g - 62, _fw_g - 10
            _now_g = time.time()
            if _bgx1 < player_x < _bgx2 and 10 < player_y < 62:
                if x_hover_start_g is None:
                    x_hover_start_g = _now_g
                _prog_g = min((_now_g - x_hover_start_g) / 1.5, 1.0)
                cv2.rectangle(frame, (_bgx1, 10), (_bgx2, 62), (0, 0, 120), -1)
                cv2.rectangle(frame, (_bgx1, int(62 - 52 * _prog_g)), (_bgx2, 62), (40, 40, 255), -1)
                if _prog_g >= 1.0:
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)
            else:
                x_hover_start_g = None
                cv2.rectangle(frame, (_bgx1, 10), (_bgx2, 62), (0, 0, 80), -1)
            cv2.rectangle(frame, (_bgx1, 10), (_bgx2, 62), (100, 100, 210), 2)
            cv2.putText(frame, "X", (_bgx1 + 14, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.imshow("Mini Jogo - Pegue as Moedas", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        salvar_pontuacao(iniciais, score)
        ranking = sorted(
            carregar_ranking().items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        x_hover_start_go = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            timestamp += 1
            go_x, go_y = detectar_dedo(frame, timestamp)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            cv2.putText(frame, "GAME OVER", (180, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(frame, f"Jogador: {iniciais}", (180, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Pontuacao: {score}", (180, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "RANKING", (220, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            y = 270
            for i, (name, points) in enumerate(ranking, 1):
                if i == 4:
                    break
                cv2.putText(frame, f"{i} - {name}: {points}", (180, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y += 35

            cv2.putText(frame, "ESC = voltar ao menu", (150, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.circle(frame, (go_x, go_y), 12, (255, 0, 0), -1)

            _fw_go = frame.shape[1]
            _bgox1, _bgox2 = _fw_go - 62, _fw_go - 10
            _now_go = time.time()
            if _bgox1 < go_x < _bgox2 and 10 < go_y < 62:
                if x_hover_start_go is None:
                    x_hover_start_go = _now_go
                _prog_go = min((_now_go - x_hover_start_go) / 1.5, 1.0)
                cv2.rectangle(frame, (_bgox1, 10), (_bgox2, 62), (0, 0, 120), -1)
                cv2.rectangle(frame, (_bgox1, int(62 - 52 * _prog_go)), (_bgox2, 62), (40, 40, 255), -1)
                if _prog_go >= 1.0:
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)
            else:
                x_hover_start_go = None
                cv2.rectangle(frame, (_bgox1, 10), (_bgox2, 62), (0, 0, 80), -1)
            cv2.rectangle(frame, (_bgox1, 10), (_bgox2, 62), (100, 100, 210), 2)
            cv2.putText(frame, "X", (_bgox1 + 14, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.imshow("Mini Jogo - Pegue as Moedas", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()