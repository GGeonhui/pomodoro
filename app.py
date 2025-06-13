# 쉬는 시간동안의 화면 수정
# 코드 전체적으로 실행 순서에 따라 정리리: 초기화 → UI 분기 → 측정/휴식 처리

# === 1. 라이브러리 import ===
import streamlit as st                       # Streamlit 웹 앱 UI 생성
import cv2                                   # OpenCV: 영상 처리용 라이브러리
import numpy as np                           # Numpy: 수치 계산용 라이브러리
from fer import FER                          # FER: 감정 인식 라이브러리
import mediapipe as mp                       # MediaPipe: 얼굴 랜드마크 추출
import pandas as pd                          # Pandas: 데이터프레임 처리
import time                                  # 시간 측정용 라이브러리
import matplotlib.pyplot as plt              # 시각화 라이브러리
from sklearn.ensemble import RandomForestRegressor  # 랜덤 포레스트 회귀 모델
import sqlite3                               # SQLite 데이터베이스 연동
import os                                    # OS 관련 함수
import winsound                              # Windows 알람 소리용

# === 2. Streamlit 페이지 설정 ===
st.set_page_config(layout="wide")
st.title("🎯 감정/집중도 기반 맞춤형 Pomodoro Timer")

# === 3. 세션 상태 초기화 ===
if 'is_measuring' not in st.session_state:
    st.session_state.is_measuring = False
if 'is_paused' not in st.session_state:
    st.session_state.is_paused = False
if 'total_elapsed_time' not in st.session_state:
    st.session_state.total_elapsed_time = 0
if 'pause_start_time' not in st.session_state:
    st.session_state.pause_start_time = 0
if 'is_break_time' not in st.session_state:
    st.session_state.is_break_time = False
if 'break_start_time' not in st.session_state:
    st.session_state.break_start_time = 0
if 'recommended_time' not in st.session_state:
    st.session_state.recommended_time = 25.0
if 'show_alarm' not in st.session_state:
    st.session_state.show_alarm = False

# === 4. 데이터베이스 초기화 ===
conn = sqlite3.connect("sessions.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''                                              
CREATE TABLE IF NOT EXISTS sessions (                           
    id INTEGER PRIMARY KEY AUTOINCREMENT,                      
    angry REAL, disgust REAL, fear REAL, happy REAL, sad REAL,  
    surprise REAL, neutral REAL,                                
    attention REAL,                                             
    recommended_minutes REAL                                   
)
''')
conn.commit()

# === 5. 함수 정의 ===
def play_alarm():
    """알람 소리 재생 함수"""
    try:
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        for _ in range(3):  # 3번 울림
            winsound.Beep(1000, 500)  # 1000Hz, 0.5초
            time.sleep(0.2)
    except:
        print("🔔 알람! 시간이 완료되었습니다!")

@st.cache_resource
def train_model():
    """머신러닝 모델 학습 함수"""
    synthetic_path = "synthetic_sessions.csv"
    if not os.path.exists(synthetic_path):
        # 기본 학습 데이터 생성
        default_data = {
            'angry': [0.1, 0.2, 0.05], 'disgust': [0.05, 0.1, 0.02],
            'fear': [0.1, 0.15, 0.08], 'happy': [0.6, 0.4, 0.7],
            'sad': [0.05, 0.1, 0.03], 'surprise': [0.05, 0.05, 0.07],
            'neutral': [0.05, 0.1, 0.05], 'attention': [0.8, 0.6, 0.9],
            'recommended_minutes': [25, 20, 30]
        }
        pd.DataFrame(default_data).to_csv(synthetic_path, index=False)
    
    df = pd.read_csv(synthetic_path)
    X = df[df.columns[:-1]]
    y = df["recommended_minutes"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# 모델 학습
model = train_model()

# === 6. 메인 UI 분기 ===

# --- 6-1. 휴식 시간 처리 ---
if st.session_state.is_break_time:
    st.markdown("## ☕ 휴식 시간")
    
    # 추천 시간 표시 (고정값)
    if 'last_recommended_time' in st.session_state:
        st.success(f"✅ 다음 세션 추천 시간: **{st.session_state.last_recommended_time}분**")
    
    break_placeholder = st.empty()  # 실시간 업데이트용 placeholder
    
    while st.session_state.is_break_time:
        current_time = time.time()
        break_elapsed = current_time - st.session_state.break_start_time
        break_remaining = 60 - break_elapsed  # 1분 = 60초
        
        if break_remaining > 0:
            minutes = int(break_remaining // 60)
            seconds = int(break_remaining % 60)
            break_placeholder.info(f"🛌 휴식 중... 남은 시간: {minutes:02d}:{seconds:02d}")
            time.sleep(1)  # 1초마다 업데이트
        else:
            # 휴식 시간 완료
            st.session_state.is_break_time = False
            st.session_state.break_start_time = 0
            if 'last_recommended_time' in st.session_state:
                st.session_state.recommended_time = st.session_state.last_recommended_time
            play_alarm()
            break_placeholder.success("🎉 휴식 시간이 완료되었습니다!")
            st.rerun()
            break

# --- 6-2. 일반 모드 (측정 준비/진행 중) ---
else:
    # 세션 시간 설정
    st.markdown("## ⌚ 세션 시간 설정")
    session_time = st.number_input(
        "측정할 세션 시간(분)", 
        min_value=0.5, 
        max_value=60.0, 
        value=st.session_state.recommended_time, 
        step=0.5
    )

    # 과거 세션 데이터 시각화
    st.subheader("📊 Previous Session Recommendation Trend")
    col_a, col_b = st.columns([4, 1])
    
    with col_a:
        df_hist = pd.read_sql_query("SELECT * FROM sessions", conn)
        if not df_hist.empty:
            fig_hist, ax_hist = plt.subplots()
            ax_hist.plot(df_hist.index + 1, df_hist["recommended_minutes"], marker='o')
            ax_hist.set_xlabel("Session Number")
            ax_hist.set_ylabel("Recommended Time (min)")
            ax_hist.set_title("Recommendation Trend")
            st.pyplot(fig_hist)
        else:
            st.info("No saved session data. Start your first measurement!")
    
    with col_b:
        if st.button("🗑️ Reset Sessions"):
            cursor.execute("DELETE FROM sessions")
            conn.commit()
            st.success("Session history has been cleared.")
            st.rerun()

    # 제어 버튼들
    st.markdown("## 🎮 세션 제어")
    button_col1, button_col2, button_col3 = st.columns(3)

    with button_col1:
        if st.button("▶ 측정 시작", disabled=st.session_state.is_measuring):
            st.session_state.is_measuring = True
            st.session_state.is_paused = False
            st.session_state.total_elapsed_time = 0
            st.session_state.pause_start_time = 0

    with button_col2:
        if st.button("⏸️ 일시정지", disabled=not st.session_state.is_measuring or st.session_state.is_paused):
            st.session_state.is_paused = True
            st.session_state.pause_start_time = time.time()

    with button_col3:
        if st.button("▶️ 재시작", disabled=not st.session_state.is_measuring or not st.session_state.is_paused):
            if st.session_state.is_paused:
                pause_duration = time.time() - st.session_state.pause_start_time
                st.session_state.total_elapsed_time += pause_duration
                st.session_state.is_paused = False
                st.session_state.pause_start_time = 0

    # 초기화 버튼
    if st.session_state.is_measuring:
        if st.button("⏹️ 초기화"):
            st.session_state.is_measuring = False
            st.session_state.is_paused = False
            st.session_state.total_elapsed_time = 0
            st.session_state.pause_start_time = 0
            st.info("초기화되었습니다.")

# === 7. 실시간 측정 처리 ===
if st.session_state.is_measuring:
    # MediaPipe 및 FER 초기화
    emotion_detector = FER(mtcnn=False)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    
    # 카메라 및 데이터 초기화
    if 'cap' not in st.session_state or st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.start_time = time.time()
        st.session_state.data = []
        st.session_state.timestamps = []
        st.session_state.attn_scores = []

    cap = st.session_state.cap
    start_time = st.session_state.start_time
    data = st.session_state.data
    timestamps = st.session_state.timestamps
    attn_scores = st.session_state.attn_scores

    # UI 레이아웃
    col1, col2 = st.columns([2, 1])
    with col1:
        frame_placeholder = st.empty()
        graph_placeholder = st.empty()
    with col2:
        emotion_placeholder = st.empty()
        timer_placeholder = st.empty()
        result_box = st.empty()

    # 실시간 측정 루프
    while st.session_state.is_measuring:
        current_time = time.time()
        
        # 경과 시간 계산 (일시정지 고려)
        if not st.session_state.is_paused:
            effective_elapsed = (current_time - start_time) - st.session_state.total_elapsed_time
        else:
            if timestamps:
                effective_elapsed = timestamps[-1]
            else:
                effective_elapsed = 0

        # 세션 시간 완료 체크
        if effective_elapsed >= session_time * 60:
            # 측정 완료 처리
            cap.release()
            cv2.destroyAllWindows()
            st.session_state.cap = None
            st.session_state.is_measuring = False
            st.session_state.is_paused = False

            # 알람 재생
            play_alarm()

            if data:
                # 데이터 분석 및 예측
                df = pd.DataFrame(data)
                avg_attention = df["attention"].mean()
                df_grouped = df[['angry','disgust','fear','happy','sad','surprise','neutral','attention']].mean().to_frame().T

                # 추천 시간 예측
                recommended_time = round(float(model.predict(df_grouped)[0]), 2)
                st.session_state.last_recommended_time = recommended_time
                
                result_box.success(f"✅ 측정 완료! 추천 시간: **{recommended_time}분**")
                st.info("🔔 1분 휴식 시간이 시작됩니다!")

                # DB 저장
                cursor.execute("""
                    INSERT INTO sessions (angry, disgust, fear, happy, sad, surprise, neutral, attention, recommended_minutes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, tuple(df_grouped.iloc[0]) + (recommended_time,))
                conn.commit()

                # CSV 저장 (학습 데이터 축적)
                synthetic_path = "synthetic_sessions.csv"
                df_grouped["recommended_minutes"] = recommended_time
                if os.path.exists(synthetic_path):
                    existing = pd.read_csv(synthetic_path)
                    updated = pd.concat([existing, df_grouped], ignore_index=True)
                    updated.to_csv(synthetic_path, index=False)
                else:
                    df_grouped.to_csv(synthetic_path, index=False)

                # 휴식 시간 시작
                st.session_state.is_break_time = True
                st.session_state.break_start_time = time.time()
            
            # 상태 초기화
            st.session_state.total_elapsed_time = 0
            st.session_state.pause_start_time = 0
            st.rerun()
            break
        
        else:
            # 측정 진행 중
            ret, frame = cap.read()
            if ret:
                # 일시정지가 아닐 때만 데이터 수집
                if not st.session_state.is_paused:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, _ = frame.shape
                    results = face_mesh.process(img_rgb)

                    # 감정 인식
                    emotions = emotion_detector.detect_emotions(frame)
                    if emotions:
                        top = emotions[0]
                        (x, y, w_box, h_box) = top["box"]
                        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                        emotion_text = ", ".join([f"{k}: {v:.2f}" for k, v in top["emotions"].items()])
                        emotion_dict = top["emotions"]
                    else:
                        emotion_text = "얼굴 감지 안됨"
                        emotion_dict = {k: 0 for k in ['angry','disgust','fear','happy','sad','surprise','neutral']}

                    # 집중도 계산
                    attention = 0
                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0]
                        points = [(lm.x * w, lm.y * h) for lm in landmarks.landmark]
                        left_eye = np.mean([points[33], points[133]], axis=0)
                        right_eye = np.mean([points[362], points[263]], axis=0)
                        eye_center = (left_eye + right_eye) / 2
                        screen_center = np.array([w / 2, h / 2])
                        dist = np.linalg.norm(eye_center - screen_center)
                        attention = max(0, 1 - dist / (w / 2))

                    # 데이터 저장
                    timestamps.append(effective_elapsed)
                    attn_scores.append(attention)
                    data.append({
                        'timestamp': effective_elapsed,
                        **emotion_dict,
                        'attention': attention
                    })

                    # 세션 상태 업데이트
                    st.session_state.data = data
                    st.session_state.timestamps = timestamps
                    st.session_state.attn_scores = attn_scores

                    emotion_placeholder.markdown(f"**감정 상태**: {emotion_text}  \n**집중도**: `{attention:.2f}`")
                else:
                    emotion_placeholder.markdown("⏸️ **일시정지 중**")

                # 영상 표시 (좌우 반전)
                frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_display = cv2.flip(frame_display, 1)
                frame_placeholder.image(frame_display, channels="RGB")

                # 집중도 그래프 업데이트
                if timestamps and attn_scores:
                    fig, ax = plt.subplots()
                    fig.set_size_inches(5, 2.5)
                    ax.plot(timestamps, attn_scores)
                    ax.set_ylim(0, 1)
                    ax.set_title("Real-time Attention")
                    graph_placeholder.pyplot(fig, use_container_width=True)

                # 타이머 표시
                remaining_time = session_time * 60 - effective_elapsed
                minutes = int(remaining_time // 60)
                seconds = int(remaining_time % 60)
                status = "⏸️ 일시정지" if st.session_state.is_paused else "▶️ 측정 중"
                timer_placeholder.markdown(f"**{status}**\n남은 시간: `{minutes:02d}:{seconds:02d}`")

            else:
                st.error("카메라 오류")
                st.session_state.is_measuring = False
                if st.session_state.cap:
                    st.session_state.cap.release()
                    st.session_state.cap = None
                break

# === 8. 카메라 정리 ===
if not st.session_state.is_measuring and 'cap' in st.session_state and st.session_state.cap is not None:
    st.session_state.cap.release()
    st.session_state.cap = None
