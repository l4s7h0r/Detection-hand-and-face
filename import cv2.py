import cv2
import mediapipe as mp

# Inicializar la detección de manos, caras y pose con MediaPipe
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Crear objetos para la detección
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands, \
     mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection, \
     mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Convertir la imagen a RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar manos
        result_hands = hands.process(image_rgb)

        # Detectar caras
        result_faces = face_detection.process(image_rgb)

        # Detectar pose (cuerpo completo)
        result_pose = pose.process(image_rgb)

        # Dibujar las manos detectadas
        if result_hands.multi_hand_landmarks:
            for hand_landmarks in result_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Dibujar las caras detectadas
        if result_faces.detections:
            for detection in result_faces.detections:
                mp_drawing.draw_detection(frame, detection)

        # Dibujar el cuerpo completo detectado
        if result_pose.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Mostrar la imagen con las detecciones
        cv2.imshow('Face, Hand, and Full Body Detection', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
