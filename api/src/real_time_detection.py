import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model with error handling
try:
    emotion_model = load_model("./emotion_recognition_model.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Initialize the webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    ).detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # If no faces detected, display message
    if len(faces) == 0:
        cv2.putText(
            frame,
            "No face detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    for x, y, w, h in faces:
        # Extract face and preprocess it
        roi_gray = gray[y : y + h, x : x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48)).reshape(1, 48, 48, 1) / 255.0

        # Debugging the shape
        print(f"Processed image shape: {roi_gray.shape}")

        # Predict emotion
        predictions = emotion_model.predict(roi_gray)

        # Debugging predictions
        print(f"Prediction: {predictions}")

        emotion = emotion_labels[np.argmax(predictions)]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2
        )

    # Display the frame
    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Quit with 'q'
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
