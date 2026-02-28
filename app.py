import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from collections import deque
import plotly.graph_objects as go

st.set_page_config(
    page_title="Face Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 100%);
    }
    .emotion-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid #0f3460;
    }
</style>
""", unsafe_allow_html=True)

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOJI = {'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨',
         'Happy': '😊', 'Neutral': '😐', 'Sad': '😢', 'Surprise': '😲'}
COLORS = {'Angry': '#ff4444', 'Disgust': '#aa00ff', 'Fear': '#ff6600',
          'Happy': '#00ff88', 'Neutral': '#888888', 'Sad': '#4488ff', 'Surprise': '#ffff00'}

@st.cache_resource
def load_emotion_model():
    return load_model('best_model.h5')

model = load_emotion_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = {e: deque(maxlen=50) for e in EMOTIONS}
if 'running' not in st.session_state:
    st.session_state.running = False
if 'dominant_emotion' not in st.session_state:
    st.session_state.dominant_emotion = 'Neutral'

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    start_btn = st.button("▶️ Start Camera", use_container_width=True)
    stop_btn = st.button("⏹️ Stop Camera", use_container_width=True)
    st.markdown("---")
    st.markdown("## 📊 Emotion Stats")
    stats_placeholder = st.empty()

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

st.markdown("<h1 style='text-align:center; color:#667eea'>😊 Face Emotion Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#888'>Real-time emotion detection using Deep Learning</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📹 Live Camera Feed")
    frame_placeholder = st.empty()

with col2:
    st.markdown("### 😊 Current Emotion")
    emotion_placeholder = st.empty()
    st.markdown("### 📈 Emotion Graph")
    graph_placeholder = st.empty()

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    frame_count = 0

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        current_emotion = 'Neutral'

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.reshape(face, (1, 48, 48, 1))

            if frame_count % 5 == 0:
                prediction = model.predict(face, verbose=0)
                emotion_idx = np.argmax(prediction)
                current_emotion = EMOTIONS[emotion_idx]
                st.session_state.dominant_emotion = current_emotion

            color = tuple(int(COLORS[current_emotion].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{current_emotion}",
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        for e in EMOTIONS:
            st.session_state.emotion_history[e].append(1 if e == current_emotion else 0)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        emotion = st.session_state.dominant_emotion
        emotion_placeholder.markdown(f"""
        <div class="emotion-box">
            <h1 style="font-size:4rem">{EMOJI.get(emotion, '😐')}</h1>
            <h2 style="color:{COLORS.get(emotion, '#fff')}">{emotion}</h2>
        </div>
        """, unsafe_allow_html=True)

        if frame_count % 10 == 0:
            fig = go.Figure()
            for e in EMOTIONS:
                if len(st.session_state.emotion_history[e]) > 0:
                    fig.add_trace(go.Scatter(
                        y=list(st.session_state.emotion_history[e]),
                        name=f"{EMOJI.get(e,'')} {e}",
                        line=dict(color=COLORS[e], width=2),
                        fill='tozeroy'
                    ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=250,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            graph_placeholder.plotly_chart(fig, use_container_width=True)

        emotion_counts = {e: sum(st.session_state.emotion_history[e]) for e in EMOTIONS}
        stats_placeholder.markdown('\n'.join([
            f"{EMOJI.get(e,'')} **{e}**: {emotion_counts[e]}" for e in EMOTIONS
        ]))

        frame_count += 1
        time.sleep(0.03)

    cap.release()
else:
    frame_placeholder.markdown("""
    <div style="background:#1a1a2e; border-radius:15px; padding:100px; text-align:center; border:2px dashed #0f3460;">
        <h2 style="color:#667eea">📹 Click 'Start Camera' to begin</h2>
        <p style="color:#888">Real-time emotion detection will appear here</p>
    </div>
    """, unsafe_allow_html=True)
