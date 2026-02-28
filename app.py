import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from collections import deque
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(
    page_title="Face Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0d0000 0%, #3d0000 30%, #7a1a00 60%, #c45000 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #3d0000 0%, #7a1a00 100%);
        border-right: 1px solid #ff6a0033;
    }
    .emotion-box {
        background: linear-gradient(135deg, #3d0000, #7a1a00);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid #ff6a00;
        box-shadow: 0 0 20px #ff6a0044;
    }
    .metric-card {
        background: linear-gradient(135deg, #3d0000, #7a1a00);
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #ff6a0033;
        margin: 5px 0;
    }
    .title-style {
        background: linear-gradient(90deg, #ff4500 0%, #ff6a00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
    }
    div[data-testid="stButton"] button {
        background: linear-gradient(90deg, #ff4500, #ff6a00);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: bold;
    }
    div[data-testid="stButton"] button:hover {
        background: linear-gradient(90deg, #ff6a00, #ff4500);
        box-shadow: 0 0 15px #ff6a00;
    }
    .stTabs [data-baseweb="tab"] {
        background: #3d0000;
        color: #ff6a00;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOJI = {'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨',
         'Happy': '😊', 'Neutral': '😐', 'Sad': '😢', 'Surprise': '😲'}
COLORS = {'Angry': '#ff4444', 'Disgust': '#aa00ff', 'Fear': '#ff6600',
          'Happy': '#00ff88', 'Neutral': '#aaaaaa', 'Sad': '#4488ff', 'Surprise': '#ffff00'}

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
if 'start_time' not in st.session_state:
    st.session_state.start_time = None

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("""
    <div style='color:#ff6a00'>
    🧠 CNN Model trained on FER2013<br><br>
    📸 61.9% Accuracy<br><br>
    👥 Multiple face detection<br><br>
    📈 Real-time emotion graph<br><br>
    🌐 Image upload support
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("## 📊 Emotion Stats")
    stats_placeholder = st.empty()

st.markdown('<p class="title-style">😊 Face Emotion Recognition</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#ff6a0088'>Real-time & Image-based emotion detection using Deep Learning</p>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2 = st.tabs(["📹 Live Camera", "📸 Upload Image"])

with tab1:
    st.markdown("### 📹 Live Camera Detection")
    st.info("⚠️ Live camera works only on local machine!")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_btn = st.button("▶️ Start Camera", use_container_width=True)
    with col_btn2:
        stop_btn = st.button("⏹️ Stop Camera", use_container_width=True)

    if start_btn:
        st.session_state.running = True
        st.session_state.start_time = time.time()
    if stop_btn:
        st.session_state.running = False
        st.session_state.start_time = None

    m1, m2, m3 = st.columns(3)
    timer_placeholder = m1.empty()
    face_placeholder = m2.empty()
    emotion_metric_placeholder = m3.empty()

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        frame_placeholder = st.empty()
    with col2:
        emotion_placeholder = st.empty()
        st.markdown("### 📈 Emotion Graph")
        graph_placeholder = st.empty()
        st.markdown("### 📊 Emotion %")
        bar_placeholder = st.empty()

    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        frame_count = 0

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            face_count = len(faces)
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

            if st.session_state.start_time:
                elapsed = int(time.time() - st.session_state.start_time)
                mins = elapsed // 60
                secs = elapsed % 60
                timer_placeholder.markdown(f"""
                <div class="metric-card" style="text-align:center">
                    <h4 style="color:#ff6a00">⏱️ Session Time</h4>
                    <h2 style="color:white">{mins:02d}:{secs:02d}</h2>
                </div>""", unsafe_allow_html=True)

            face_placeholder.markdown(f"""
            <div class="metric-card" style="text-align:center">
                <h4 style="color:#ff6a00">👥 Faces Detected</h4>
                <h2 style="color:white">{face_count}</h2>
            </div>""", unsafe_allow_html=True)

            emotion_metric_placeholder.markdown(f"""
            <div class="metric-card" style="text-align:center">
                <h4 style="color:#ff6a00">😊 Dominant Emotion</h4>
                <h2 style="color:{COLORS.get(emotion, '#fff')}">{EMOJI.get(emotion,'')} {emotion}</h2>
            </div>""", unsafe_allow_html=True)

            emotion_placeholder.markdown(f"""
            <div class="emotion-box">
                <h1 style="font-size:4rem">{EMOJI.get(emotion, '😐')}</h1>
                <h2 style="color:{COLORS.get(emotion, '#fff')}">{emotion}</h2>
            </div>""", unsafe_allow_html=True)

            if frame_count % 10 == 0:
                fig = go.Figure()
                for e in EMOTIONS:
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
                    height=200,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=True,
                    legend=dict(font=dict(size=8))
                )
                graph_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_{frame_count}")

                total = sum(sum(st.session_state.emotion_history[e]) for e in EMOTIONS)
                if total > 0:
                    bar_fig = go.Figure()
                    for e in EMOTIONS:
                        pct = (sum(st.session_state.emotion_history[e]) / total) * 100
                        bar_fig.add_trace(go.Bar(
                            x=[pct], y=[e], orientation='h',
                            name=e, marker_color=COLORS[e]
                        ))
                    bar_fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        height=200,
                        margin=dict(l=0, r=0, t=0, b=0),
                        showlegend=False
                    )
                    bar_placeholder.plotly_chart(bar_fig, use_container_width=True, key=f"bar_{frame_count}")

            emotion_counts = {e: sum(st.session_state.emotion_history[e]) for e in EMOTIONS}
            stats_placeholder.markdown('\n'.join([
                f"{EMOJI.get(e,'')} **{e}**: {emotion_counts[e]}" for e in EMOTIONS
            ]))

            frame_count += 1
            time.sleep(0.03)

        cap.release()
    else:
        frame_placeholder.markdown("""
        <div style="background:linear-gradient(135deg, #3d0000, #7a1a00); border-radius:15px; padding:80px; text-align:center; border:2px dashed #ff6a00;">
            <h2 style="color:#ff6a00">📹 Click 'Start Camera' to begin</h2>
            <p style="color:#888">Real-time emotion detection will appear here</p>
        </div>""", unsafe_allow_html=True)

with tab2:
    st.markdown("### 📸 Upload Image Detection")
    st.success("✅ Works on Streamlit Cloud & Local!")

    uploaded_file = st.file_uploader("Upload a face image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Original Image")
            image_resized = image.resize((300, 300))
            st.image(image_resized, width=300)

        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        with col2:
            st.markdown("#### Detected Emotions")
            if len(faces) == 0:
                st.warning("😕 No face detected! Please upload a clear face image.")
            else:
                result_img = img_array.copy()
                emotions_detected = []

                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, (48, 48))
                    face_normalized = face_resized / 255.0
                    face_input = np.reshape(face_normalized, (1, 48, 48, 1))

                    prediction = model.predict(face_input, verbose=0)
                    emotion_idx = np.argmax(prediction)
                    detected_emotion = EMOTIONS[emotion_idx]
                    emotions_detected.append((detected_emotion, prediction))

                    color = tuple(int(COLORS[detected_emotion].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(result_img, detected_emotion,
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                result_img_resized = cv2.resize(result_img, (300, 300))
                st.image(result_img_resized, width=300)

                for idx, (detected_emotion, prediction) in enumerate(emotions_detected):
                    confidence = prediction[0][EMOTIONS.index(detected_emotion)] * 100
                    st.markdown(f"""
                    <div class="emotion-box">
                        <h1>{EMOJI.get(detected_emotion, '😐')}</h1>
                        <h2 style="color:{COLORS.get(detected_emotion, '#fff')}">{detected_emotion}</h2>
                        <p style="color:#ff6a00">Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                    conf_fig = go.Figure()
                    for i, e in enumerate(EMOTIONS):
                        conf_fig.add_trace(go.Bar(
                            x=[prediction[0][i] * 100],
                            y=[f"{EMOJI.get(e,'')} {e}"],
                            orientation='h',
                            marker_color=COLORS[e]
                        ))
                    conf_fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        height=250,
                        margin=dict(l=0, r=0, t=0, b=0),
                        showlegend=False,
                        xaxis=dict(range=[0, 100], title="Confidence %")
                    )
                    st.plotly_chart(conf_fig, use_container_width=True, key=f"conf_{idx}")
