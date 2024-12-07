import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
try:
    emotion_model = load_model('D:/face/final_emotion_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar cascade for face detection
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)
# Initialize webcam


if not cap.isOpened():
    print("Error: Unable to access the external webcam.")
    cap = cv2.VideoCapture(0)
else:
    print("Internal webcam accessed successfully.")

print("Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture an image.")
            break

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48))

        for (x, y, w, h) in faces:
            # Extract and preprocess face ROI
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype('float32') / 255.0  # Normalize pixel values
            roi_gray = np.expand_dims(roi_gray, axis=[0, -1])  # Add batch and channel dimensions

            # Predict emotion
            predictions = emotion_model.predict(roi_gray, verbose=0)
            max_index = np.argmax(predictions)
            emotion = emotion_labels[max_index]
            emotion_confidence = predictions[0][max_index] * 100

            # Draw rectangle and emotion label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{emotion} ({emotion_confidence:.1f}%)", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                        (255, 255, 255), 2)

        # Show frame
        cv2.imshow('Real-Time Emotion Detection', frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
