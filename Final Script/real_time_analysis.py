import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the saved model
model = load_model('Models/facial_emotion_detection_model.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize OpenCV's Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Preprocess the face region for emotion prediction
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # Predict emotion
        prediction = model.predict(roi)[0]
        max_index = int(np.argmax(prediction))
        emotion = emotion_labels[max_index]
        
        # Display the emotion on the frame
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    # Display the resulting frame
    cv2.imshow('Real-time Emotion Analysis', frame)
    
    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
