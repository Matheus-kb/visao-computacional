import os
import cv2
from deepface import DeepFace
import numpy as np

# Carregar o modelo Emotion do DeepFace
model = DeepFace.build_model("Emotion")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']