import os
import cv2
from deepface import DeepFace
import numpy as np

# Carregar o modelo Emotion do DeepFace
model = DeepFace.build_model("Emotion")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Carregar o classificador de faces
face_cascade = cv2.CascadeClassifier('c:\\Projetos\\visao-computacional\\analise-de-sentimento-facial\\haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)  # Para usar a câmera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar imagem da câmera. Verifique a conexão da câmera.")
        break
    
    # Converter o frame para escala de cinza
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y + h, x:x + w]
        
        # Redimensionar e normalizar a face
        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
        normalized_face = resized_face / 255.0
        
        # Converter a profundidade da imagem para 32 bits (CV_32F)
        normalized_face = normalized_face.astype(np.float32)
        
        # Verificar a forma e canais da imagem antes de predizer
        reshaped_face = normalized_face.reshape(1, 48, 48, 1)
        print("Formato da imagem antes da predição:", reshaped_face.shape)
        
        try:
            # Converter a imagem para três canais (RGB)
            if reshaped_face.shape[-1] == 1:
                reshaped_face_rgb = np.repeat(reshaped_face, 3, axis=-1)
            else:
                reshaped_face_rgb = reshaped_face
            
            # Fazer a predição
            preds = model.predict(reshaped_face_rgb)
            print("Predições do modelo:", preds)
        except Exception as e:
            print("Erro durante a predição:", e)
            continue
        
        # Obter o índice da emoção prevista
        emotion_idx = np.argmax(preds)
        emotion = emotion_labels[emotion_idx]
        
        # Desenhar o retângulo ao redor da face e a emoção prevista
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Mostrar o frame com as predições
    cv2.imshow('Real-time Emotion Detection', frame)
    
    # Parar o loop ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e destruir todas as janelas
cap.release()
cv2.destroyAllWindows()

# "-" 
