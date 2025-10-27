Emotion Recognition from Human Pose using Random Forest
Project Overview

This project focuses on detecting human emotions from video input or live camera streams by analyzing body posture and movement using MediaPipe Pose Estimation and Machine Learning (Random Forest Classifier).
The system extracts 3D pose landmarks from video frames, processes them into numerical features, scales them, and predicts the most likely emotion class.

Workflow Summary
1. Input

Video file (.mp4) or camera stream captured in real-time

Example: testing.mp4

2. Process

Extracts pose keypoints using MediaPipe Pose

Selects significant joint coordinates (x, y, z)

Scales the features using a pre-trained StandardScaler

Classifies emotion using a Random Forest model

3. Output

Displays the video with the predicted emotion label overlaid on each frame.

📂 Folder Structure
📁 BPM-GCN/
│
├── data/                     # Folder to be uploaded manually in Colab session
│   ├── train/                # Contains training video samples
│   ├── test/                 # Contains testing video samples
│
├── emotion_classifier_rf.pkl # Saved Random Forest model
├── scaler.pkl                # Scaler used for feature normalization
├── testing.mp4               # Sample video for prediction
├── main.ipynb                # Colab notebook for running the project
└── README.md                 # Project documentation


Note: The data/ folder will not be included in this repository.
It must be uploaded manually in your Google Colab local session before training or testing the model.

Setup Instructions
Step 1 — Clone the Repository
!git clone https://github.com/hyper-maniac/BPM-GCN.git
%cd BPM-GCN

Step 2 — Upload Training Data

In your Colab session
create a folder named data, and upload the data folder manually using:

from google.colab import files
files.upload()


Or use the file upload option in Colab’s sidebar.

Step 3 — Train the Model

Run the training cells:

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Split and train model
# Save model and scaler

Step 4 — Test the Model
import cv2
import mediapipe as mp
import joblib

# Load model and process the video

Step 5 — View the Results

Processed video frames with the predicted emotion label will be displayed using cv2_imshow().

Technologies Used

Python 3

Google Colab

scikit-learn

MediaPipe

OpenCV

NumPy

Joblib
