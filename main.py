import cv2, time

# Função para abrir a câmera testando múltiplos índices
def open_camera(indices=(0, 1, 2)):
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap, idx
            cap.release()
    return None, None

# Carrega classificador pré-treinado de detecção facial
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Inicializa webcam
cap, cam_idx = open_camera()
start_time = time.time()

if cap is None:
    print("Erro ao acessar a camera. Indices testados: 0, 1, 2")
    raise SystemExit(1)

print(f"Camera ativa no indice {cam_idx}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converte para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta rostos
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    cv2.putText(frame, f"Pessoas: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2)
    fps = 1.0 / (time.time() - start_time)
    start_time = time.time()

    cv2.putText(frame, f"FPS: {int(fps)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2)

    # Desenha retângulo ao redor do rosto
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Rosto Detectado", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

    cv2.imshow("Reconhecimento Facial - Mostra CC", frame)

    # Pressione ESC para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
