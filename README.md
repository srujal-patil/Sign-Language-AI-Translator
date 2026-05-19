# Real-Time Sign Language Recognition Using Computer Vision and Machine Learning

## 📋 Project Overview
This project is an end-to-end, real-time sign language recognition system designed to bridge the communication gap between the hearing/speech-impaired community and the general public. By utilizing state-of-the-art computer vision algorithms and machine learning classifiers, the system captures real-time video from a standard webcam, tracks complex hand geometry, and translates static hand gestures into textual alphabet letters on screen instantly.

---

## 🧠 Model Download & Setup
Because the trained AI model is highly detailed, the serialized brain file exceeds GitHub's standard upload limits. 

1. **Download the Model:** Download `isl_trained_model.p` from my Google Drive link here: **https://drive.google.com/file/d/1JqjwcxZ9TbwHLCuRVLjI-oqntHy4zKIE/view?usp=sharing**
2. **Placement:** Move the downloaded `isl_trained_model.p` file directly into your local project root folder right next to `test_webcam.py`.

---

## ✨ Key Features
* **Real-Time Landmark Detection:** Captures and visually maps 21 individual hand joint coordinates in 3D space ($X, Y, Z$) simultaneously.
* **High-Accuracy Classification:** Employs a robust machine learning backend optimized for coordinate data, yielding precise gesture mapping.
* **Robust Image Pipeline:** Includes automatic preprocessing capable of rendering color and grayscale image variations uniformly to prevent runtime exceptions.
* **Lightweight Deployment:** Operates smoothly on consumer-grade laptop webcams without requiring heavy hardware resources.

---

## 🛠️ Technical Architecture & Methodology

The project structure is split into three distinct developmental phases:

### 1. Data Extraction & Preprocessing (The Feature Engine)
* **Framework:** MediaPipe Tasks API (Vision Hand Landmarker)
* **Process:** The system targets a raw image dataset containing thousands of sign language letters. Each image is dynamically preprocessed into a 3-channel RGB image format. The computer vision model extracts the spatial geometry of up to two hands per frame.
* **Output:** Every image is compressed into a flat array of 126 mathematical landmarks (2 hands × 21 joints × 3 dimensions) and compiled into a structured dataset (`isl_dataset_cleaned.csv`).

### 2. Model Training (The AI Brain)
* **Framework:** Scikit-Learn (Python)
* **Algorithm:** Random Forest Classifier (100 Decision Tree estimators)
* **Process:** The coordinate dataset was split into an 80% training set and a 20% stratified testing set. The Random Forest model mapped the high-dimensional spatial coordinates against their corresponding alphabetical labels.
* **Output:** A serialized predictive brain file (`isl_trained_model.p`) capable of evaluating new coordinate arrays in fractions of a millisecond.

### 3. Live Webcam Deployment (The Interface)
* **Framework:** OpenCV & MediaPipe
* **Process:** The interface establishes a live camera stream, captures real-time hand movements, maps the mathematical skeletons on the screen, feeds the live arrays directly to the trained model, and projects the resulting translation onto the active display buffer.

---

## 📊 System Performance & Results

* **Machine Learning Backend:** Random Forest Classifier
* **Validation Metric:** Stratified Hold-Out Test Split
* **Final Model Accuracy:** **99.64%**

The exceptional accuracy score validates that structural hand skeleton profiles offer a highly stable representation for gesture recognition compared to raw, pixel-heavy convolutional image networks.

---

## 💻 Technology Stack
* **Language:** Python 3.11
* **Computer Vision:** Google MediaPipe (Tasks API), OpenCV (`cv2`)
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`sklearn`)
* **Environment:** Google Colab (Cloud Training), Visual Studio Code (Local Execution)

---

## 🚀 How to Run Locally

1. Clone or download this repository.
2. Ensure you have downloaded the `isl_trained_model.p` file from the Google Drive link above and placed it in this directory.
3. Install the required dependencies using your terminal:
   ```bash
   pip install -r requirements.txt
4. Launch the live webcam translator application:
   ```bash
   python test_webcam.py
5. Press 'q' on your keyboard while focusing on the webcam window to exit the application.
