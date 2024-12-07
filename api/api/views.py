import base64
import os
import cv2
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from tensorflow.keras.models import load_model
from .serializers import ImageUploadSerializer

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the model file
model_path = os.path.join(current_dir, "final_emotion_model.h5")  

# Load the trained model
try:
    emotion_model = load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

class EmotionDetectionView(APIView):
    def post(self, request, *args, **kwargs):
        image_data = request.data.get("image")
        if not image_data:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            image_bytes = base64.b64decode(image_data)
            image_array = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            ).detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48))

            if len(faces) == 0:
                return Response(
                    {"emotion": "No face detected"}, status=status.HTTP_200_OK
                )

            for (x, y, w, h) in faces:
                roi_gray = gray_frame[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype('float32') / 255.0  # Normalize pixel values
                roi_gray = np.expand_dims(roi_gray, axis=[0, -1])  # Add batch and channel dimensions

                predictions = emotion_model.predict(roi_gray, verbose=0)
                max_index = np.argmax(predictions)
                emotion = emotion_labels[max_index]
                emotion_confidence = predictions[0][max_index] * 100

                return Response({"emotion": emotion, "confidence": emotion_confidence}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)