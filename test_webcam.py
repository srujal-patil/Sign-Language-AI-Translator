import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
import pickle
import os
import urllib.request

# 1. Load your newly trained AI brain
model_filename = 'isl_trained_model.p'
if not os.path.exists(model_filename):
    print(f"Error: Cannot find '{model_filename}' in this folder! Make sure you moved it here.")
    exit()

with open(model_filename, 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']

# 2. Download the tracking model locally if it's missing
model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand tracking model onto your PC... please wait...")
    urllib.request.urlretrieve(model_url, model_path)

# 3. Initialize the Modern MediaPipe Detector
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# 4. Turn on your Webcam
cap = cv2.VideoCapture(0)
print("Webcam successfully started! Press 'q' on your keyboard to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally so it acts like a mirror
    frame = cv2.flip(frame, 1)
    
    # Convert colors (OpenCV uses BGR, MediaPipe needs RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Wrap frame in MediaPipe's specialized image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # Run the AI hand detector
    detection_result = detector.detect(mp_image)
    
    landmarks_data = np.zeros(126)
    
    if detection_result.hand_landmarks:
        for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
            if i > 1: break # Max 2 hands
            
            hand_coords = []
            for lm in hand_landmarks:
                hand_coords.extend([lm.x, lm.y, lm.z])
                
                # Draw visual dots on your hand on the screen
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                
            start_idx = i * 63
            end_idx = start_idx + 63
            landmarks_data[start_idx:end_idx] = hand_coords
        
        # Ask your trained AI brain to guess the letter based on the coordinates
        prediction = model.predict([landmarks_data])
        predicted_letter = prediction[0]
        
        # Draw the predicted letter on your screen in bright blue text
        cv2.putText(frame, f"Letter: {predicted_letter}", (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

    # Show the window
    cv2.imshow('Indian Sign Language AI Translator', frame)
    
    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()