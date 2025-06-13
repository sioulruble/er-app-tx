import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
import random
import pandas as pd
import os
from datetime import datetime
import sys
import traceback

try:

    import sys
    import os
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_dir)

    try:
        from lstm_model import LSTMEmotionModel
        LSTM_AVAILABLE = True
    except ImportError:
        try:
            sys.path.append(os.path.join(project_dir, 'models'))
            from lstm.lstm_model import LSTMEmotionModel
            LSTM_AVAILABLE = True
        except ImportError:

            try:
                sys.path.append(os.path.join(project_dir, 'models', 'lstm'))
                from lstm_model import LSTMEmotionModel
                LSTM_AVAILABLE = True
            except ImportError:
                st.error("LSTM module not found. Please check the file structure.")
                LSTM_AVAILABLE = False
except Exception as e:
    st.error(f"Error importing LSTM module: {str(e)}")
    LSTM_AVAILABLE = False

def main():
    st.title("TX - Emotion Recognition 😀")
    st.markdown("""
    This application performs real-time emotion recognition from all the models made by the students in this TX project.  
    It can process content from your webcam or uploaded files, and detect emotions like happiness, sadness, anger, surprise, fear, disgust, and neutral.
    Use the sidebar to select your preferred model and input source.
    """)

    st.sidebar.title("Options")

    if LSTM_AVAILABLE:
        emotion_model = st.sidebar.selectbox(
            "Choose emotion recognition model",
            ["LSTM Model"]
        )
    else:
        emotion_model = st.sidebar.selectbox(
            "Choose emotion recognition model",
            ["Basic Model", "Advanced Model", "Custom Model"]
        )
        if emotion_model == "LSTM Model":
            st.error("LSTM Model is not available. Please check the installation.")
    
    media_type = st.sidebar.radio("Media Type", ["Image", "Video"])

    if emotion_model == "LSTM Model" and media_type == "Image":
        st.warning("LSTM model is designed for video analysis only. Results may not be accurate for single images.")
    
    source_type = st.sidebar.radio("Source", ["Gallery", "Webcam"])
    
    if media_type == "Image":
        process_image(source_type, emotion_model)
    else:
        process_video(source_type, emotion_model)

def process_image(source_type, emotion_model):
    if source_type == "Gallery":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            if st.button("Analyze Emotions"):
                with st.spinner("Analyzing..."):
                    result = analyze_emotion(img_array, emotion_model)
                    annotated_image = annotate_image(img_array, result)
                    st.image(annotated_image, caption="Analyzed Image", use_column_width=True)
    else:
        st.info("Click 'Take Picture' to capture from webcam")
        if st.button("Take Picture"):
            picture = st.camera_input("Webcam")
            if picture:
                image = Image.open(picture)
                img_array = np.array(image)
                with st.spinner("Analyzing..."):
                    result = analyze_emotion(img_array, emotion_model)
                    annotated_image = annotate_image(img_array, result)
                    st.image(annotated_image, caption="Analyzed Image", use_column_width=True)

def process_video(source_type, emotion_model):
    if source_type == "Gallery":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            st.video(uploaded_file)
            
            if st.button("Analyze Emotions"):
                with st.spinner("Analyzing video..."):
                    csv_data = analyze_video_to_csv(tfile.name, emotion_model)
                    st.success("Analysis complete!")
                    col1, col2 = st.columns(2)
                    with col1:
                        csv = csv_data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"emotions_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    with col2:
                        st.dataframe(csv_data)
    else:
        if 'recording' not in st.session_state:
            st.session_state.recording = False
            st.session_state.emotion_data = []
            st.session_state.frames = []
            st.session_state.start_time = 0
        
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.recording:
                if st.button("Start Recording"):
                    st.session_state.recording = True
                    st.session_state.emotion_data = []
                    st.session_state.frames = []
                    st.session_state.start_time = time.time()
        with col2:
            if st.session_state.recording:
                if st.button("Stop Recording"):
                    st.session_state.recording = False
        
        stframe = st.empty()
        
        if st.session_state.recording:
            stframe.info("Recording in progress... Please wait.")
            placeholder = st.empty()
            
            cap = cv2.VideoCapture(1)
            try:
                frame_count = 0
                while st.session_state.recording:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Webcam disconnected")
                        st.session_state.recording = False
                        break
                    
                    current_time = time.time() - st.session_state.start_time
                    timestamp = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    
                    if frame_count % 2 == 0:
                        result = analyze_emotion(frame, emotion_model)
                        
                        emotions = list(result["emotion"].keys())
                        scores = list(result["emotion"].values())
                        dominant_emotion = emotions[np.argmax(scores)]
                        
                        st.session_state.emotion_data.append({
                            'timestamp': timestamp,
                            'seconds': round(current_time, 3),
                            'emotion': dominant_emotion,
                            'confidence': max(scores)
                        })
                        
                        st.session_state.frames.append(frame.copy())
                        
                        annotated_frame = annotate_image(frame, result)
                        placeholder.image(annotated_frame, channels="BGR", caption=f"Recording: {int(current_time)}s")
                    
                    frame_count += 1
                    time.sleep(0.05)
            
            finally:
                cap.release()
        
        if not st.session_state.recording and len(st.session_state.emotion_data) > 0:
            st.success("Recording complete! Download files below.")
            
            csv_df = pd.DataFrame(st.session_state.emotion_data)
            col1, col2 = st.columns(2)
            with col1:
                csv = csv_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Emotions CSV",
                    data=csv,
                    file_name=f"emotions_webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            with col2:
                st.dataframe(csv_df)
            
            if len(st.session_state.frames) > 0:
                with st.spinner("Generating video file..."):
                    try:
                        height, width, _ = st.session_state.frames[0].shape
                        video_path = tempfile.mktemp(suffix='.mp4')
                        
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
                        
                        for frame in st.session_state.frames:
                            out.write(frame)
                        
                        out.release()
                        
                        with open(video_path, 'rb') as f:
                            video_bytes = f.read()
                        
                        st.download_button(
                            label="Download Video",
                            data=video_bytes,
                            file_name=f"video_webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                            mime="video/mp4"
                        )
                    except Exception as e:
                        st.error(f"Error generating video: {str(e)}")

def analyze_emotion(image, model_type, real_time=False):
    if model_type == "LSTM Model" and LSTM_AVAILABLE:
        try:

            lstm_model = LSTMEmotionModel()
            return lstm_model.analyze_frame(image)
        except Exception as e:
            st.error(f"Error analyzing with LSTM model: {str(e)}")
            traceback.print_exc()
            return default_emotion_analysis()
    else:

        return default_emotion_analysis()

def default_emotion_analysis():
    emotions = ["happy", "sad", "angry", "surprised", "fearful", "disgusted", "neutral"]
    
    result = {"emotion": {}}
    
    for emotion in emotions:
        result["emotion"][emotion] = random.random()
    
    total = sum(result["emotion"].values())
    for emotion in result["emotion"]:
        result["emotion"][emotion] /= total
    
    return result

def annotate_image(image, result):
    img_copy = image.copy()
    
    emotions = list(result["emotion"].keys())
    scores = list(result["emotion"].values())
    dominant_emotion = emotions[np.argmax(scores)]
    confidence = max(scores)
    
    text = f"{dominant_emotion.upper()}: {confidence:.2f}"
    
    height, width = img_copy.shape[:2]
    
    text_position = (10, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    cv2.rectangle(img_copy, (5, 5), (text_size[0] + 15, 40), (0, 0, 0), -1)
    
    cv2.putText(img_copy, text, text_position, font, font_scale, (255, 255, 255), font_thickness)
    
    return img_copy

def analyze_video_to_csv(video_path, model_type):
    if model_type == "LSTM Model" and LSTM_AVAILABLE:
        try:
            lstm_model = LSTMEmotionModel()
            return lstm_model.process_video(video_path)
        except Exception as e:
            st.error(f"Error analyzing video with LSTM model: {str(e)}")
            traceback.print_exc()
            return default_video_analysis(video_path)
    else:
        return default_video_analysis(video_path)

def default_video_analysis(video_path):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    emotion_data = []
    
    with st.progress(0) as progress_bar:
        for frame_count in range(total_frames):
            if frame_count % 5 == 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_count / fps
                
                result = default_emotion_analysis()
                
                emotions = list(result["emotion"].keys())
                scores = list(result["emotion"].values())
                dominant_emotion = emotions[np.argmax(scores)]
                
                emotion_data.append({
                    'frame': frame_count,
                    'timestamp': timestamp,
                    'emotion': dominant_emotion,
                    'confidence': max(scores)
                })
            if progress_bar:
        
                progress_bar.progress(min(frame_count / total_frames, 1.0))
    
    cap.release()
    
    csv_df = pd.DataFrame(emotion_data)
    return csv_df

if __name__ == "__main__":
    st.set_page_config(
        page_title="TX - Emotion Recognition",
        page_icon="😀",
        layout="wide"
    )
    main()