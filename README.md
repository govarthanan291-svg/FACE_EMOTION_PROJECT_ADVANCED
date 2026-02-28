# Face Emotion Recognition - Advanced 😊🌅

A real-time face emotion detection web application built with Streamlit and Deep Learning.

## 🎯 About
This is the advanced version of the Face Emotion Recognition project. It features a beautiful Sunset themed UI with both live camera detection and image upload support.

## ⚠️ Important Note
> **Live Camera Tab** - Works only on Local Machine (requires webcam access)
> 
> **Image Upload Tab** - Works on both Local Machine and Streamlit Cloud ✅

## 🧠 Model Details
- Architecture: Custom CNN (Convolutional Neural Network)
- Dataset: FER2013 (35,887 images)
- Accuracy: ~61.9%
- Training: Google Colab (GPU)
- Epochs: 43 (Early Stopping)

## 😶 Emotions Detected
- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

## 🛠️ Technologies Used
- Python 3.10
- TensorFlow / Keras
- OpenCV
- Streamlit
- Plotly
- NumPy
- PIL

## 📦 Installation

### Clone the repository
```
git clone https://github.com/govarthanan291-svg/FACE_EMOTION_PROJECT_ADVANCED.git
cd FACE_EMOTION_PROJECT_ADVANCED
```

### Create conda environment
```
conda create -n emotion_env python=3.10
conda activate emotion_env
```

### Install dependencies
```
pip install -r requirements.txt
```

## 🚀 Usage
```
streamlit run app.py
```

## 🌐 Features
- 📹 **Live Camera Tab** - Real-time emotion detection (Local only)
- 📸 **Image Upload Tab** - Upload any face image and detect emotion
- 📊 Real-time emotion graph
- 📈 Emotion percentage bar chart
- ⏱️ Session timer
- 👥 Multiple face detection
- 🌅 Beautiful Sunset theme UI

## 📸 Screenshots

### 📹 Live Camera Detection

### 😊 Happy
![Happy](happy%202.png)

### 😢 Sad
![Sad](sad%202.png)

### 🤢 Disgust
![Disgust](disgust%202.png)

### 📸 Image Upload Detection

### 😠 Angry Upload
![Angry Upload](angry%202.png)

### 😊 Happy Upload
![Happy Upload](upload%20happy%202.png)

## 👨‍💻 Author
Govarthanan B

## 🔗 Links
- [Basic Version](https://github.com/govarthanan291-svg/face_emotion_project_basic)
- [Intermediate Version](https://github.com/govarthanan291-svg/face_emotion_project_intermediate)
