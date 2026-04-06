import cv2, time

# Carrega classificador pré-treinado de detecção facial
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Inicializa webcam
cap = cv2.VideoCapture(3)
start_time = time.time()

if not cap.isOpened():
    print("Erro ao acessar a câmera")
    exit()

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
