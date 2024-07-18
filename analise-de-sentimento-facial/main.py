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