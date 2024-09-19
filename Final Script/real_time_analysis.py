import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('Models/emotion_detection_model.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize MediaPipe Face Mesh for face landmark detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Gamma correction function
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Start capturing video
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Apply gamma correction to handle low-light conditions
    gamma_corrected_frame = adjust_gamma(frame, gamma=1.5)  # Increase brightness with gamma > 1.0
    
    # Convert to RGB as Face Mesh expects RGB input
    rgb_frame = cv2.cvtColor(gamma_corrected_frame, cv2.COLOR_BGR2RGB)
    
    # Perform face landmarks detection
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract the face landmarks (468 points)
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y])
            
            # Convert the landmarks into a NumPy array
            landmarks = np.array(landmarks, dtype=np.float32)
            
            # Reshape the landmarks into (468, 2, 1) to match model input
            landmarks = np.expand_dims(landmarks, axis=-1)  # Add the channel dimension (468, 2, 1)
            landmarks = np.expand_dims(landmarks, axis=0)  # Add the batch dimension (1, 468, 2, 1)

            # Predict emotion
            prediction = model.predict(landmarks)[0]
            max_index = int(np.argmax(prediction))
            emotion = emotion_labels[max_index]

            # Get the face bounding box to display the emotion label
            h, w, c = frame.shape
            x_min = int(min([landmark.x for landmark in face_landmarks.landmark]) * w)
            y_min = int(min([landmark.y for landmark in face_landmarks.landmark]) * h)
            x_max = int(max([landmark.x for landmark in face_landmarks.landmark]) * w)
            y_max = int(max([landmark.y for landmark in face_landmarks.landmark]) * h)

            # Display the emotion on the frame
            cv2.putText(frame, emotion, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Real-time Emotion Analysis', frame)
    
    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
