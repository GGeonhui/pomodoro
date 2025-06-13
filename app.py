
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
from datetime import datetime                # 날짜 시간 처리용
import signal                                # 프로그램 종료용

# === 2. Streamlit 페이지 설정 ===
st.set_page_config(layout="wide")

# 제목과 종료 버튼을 한 줄에 배치
title_col, exit_col = st.columns([5, 1])
with title_col:
    st.title("🎯 감정/집중도 기반 맞춤형 Pomodoro Timer")
with exit_col:
    if st.button("🚪 타이머 중지(종료)", type="secondary"):
        # 측정 중이면 카메라 해제
        if 'cap' in st.session_state and st.session_state.cap is not None:
            st.session_state.cap.release()
            cv2.destroyAllWindows()
        
        st.error("🚪 프로그램 종료 중...")
        st.info("터미널에서 Ctrl+C를 눌러 완전히 종료하거나, 브라우저 탭을 닫아주세요.")
        
        # 프로세스 종료 시도
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except:
            st.stop()

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
    recommended_minutes REAL,
    session_date TEXT,
    session_duration REAL
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

def generate_synthetic_data():
    """개선된 가상 데이터 생성 함수 (data.py 로직 적용)"""
    np.random.seed(42)
    n = 200
    
    # 베타분포를 사용한 현실적인 감정 분포
    df = pd.DataFrame({
        "angry": np.random.beta(3, 4, n),            # 매우 집중할 때 높아짐 (중간~높은 값 분포)
        "disgust": np.random.beta(1, 9, n),          # 기본적으로 낮게 유지 (거의 0에 가까움)
        "fear": np.random.beta(3, 4, n),             # 매우 집중할 때 높아짐 (중간~높은 값 분포)
        "happy": np.random.beta(1, 9, n),            # 기본적으로 낮게 유지 (거의 0에 가까움)
        "sad": np.random.beta(3, 4, n),              # 매우 집중할 때 높아짐 (중간~높은 값 분포)
        "surprise": np.random.beta(1, 9, n),         # 기본적으로 낮게 유지 (거의 0에 가까움)
        "neutral": np.random.beta(4, 3, n),          # 보통 집중할 때 높아짐 (높은 값 분포)
        "attention": np.random.uniform(0.3, 0.95, n) # 집중도 범위 (산만:0.3~0.5, 보통:0.5~0.7, 높음:0.7~0.8)
    })
    
    # 개선된 가중치 (data.py에서 가져온 값)
    weights = np.array([
        1.1,    # angry: 증가
        0.0,    # disgust: 의미 없음
        1.1,    # fear: 증가
        0.0,   # happy: 감소
        1.1,    # sad: 증가
        0.0,   # surprise: 의미 없음
        -0.5,   # neutral: 보통 집중 → 시간 감소 (더 높은 집중 유도)
        0.8     # attention: 시선 집중도
    ])
    
    # 노이즈 추가
    noise = np.random.normal(0, 1.5, n)
    
    # 추천 시간 계산
    df["recommended_minutes"] = 25 + df[df.columns].values @ weights + noise
    
    # 범위 조정 (20-50분)
    df["recommended_minutes"] = np.clip(df["recommended_minutes"], 20, 50)
    
    # 특별한 경우 처리: attention이 매우 낮으면 30분 이상
    low_attention_mask = df["attention"] < 0.5
    df.loc[low_attention_mask, "recommended_minutes"] = np.clip(
        df.loc[low_attention_mask, "recommended_minutes"], 30, 50
    )
    
    return df

def calculate_recommendation_with_improved_logic(df_grouped):
    """개선된 추천 로직 (session_core.py 로직 적용)"""
    # 집중 상태 분석
    angry_score = df_grouped['angry'].iloc[0]
    neutral_score = df_grouped['neutral'].iloc[0]
    attention_score = df_grouped['attention'].iloc[0]
    
    # 집중 상태별 시간 추천 로직
    if angry_score > 0.12 and attention_score >= 0.5:  # 매우 집중 상태
        recommended_time = 25 + angry_score * 15 + attention_score * 10
        status = f"매우 집중 (Neutral: {neutral_score:.2f}, Angry: {angry_score:.2f}, Attention: {attention_score:.2f}) → 집중력 유지하며 시간 증가"
    elif neutral_score >= 0.6 and attention_score >= 0.5: # 보통 집중 상태
        recommended_time = max(20, 25 - neutral_score * 8)
        status = f"보통 집중 (Neutral: {neutral_score:.2f}, Angry: {angry_score:.2f}, Attention: {attention_score:.2f}) → 더 높은 집중 유도를 위해 시간 단축"
    else:  # 산만한 상태
        recommended_time = 30.0
        status = f"산만함 (Neutral: {neutral_score:.2f}, Angry: {angry_score:.2f}, Attention: {attention_score:.2f}) → 차분히 앉아있기 위해 긴 시간 권장"
    
    # 추천 시간 범위 제한 (20-50분)
    recommended_time = np.clip(recommended_time, 20, 50)
    
    return recommended_time, status

@st.cache_resource
def train_model():
    """머신러닝 모델 학습 함수 (개선된 데이터 사용)"""
    synthetic_path = "synthetic_sessions.csv"
    if not os.path.exists(synthetic_path):
        # 개선된 학습 데이터 생성
        df = generate_synthetic_data()
        df.to_csv(synthetic_path, index=False)
        st.info("🔄 개선된 학습 데이터가 생성되었습니다.")
    
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
    
    # 추천 시간과 상태 표시
    if 'last_recommended_time' in st.session_state:
        st.success(f"✅ 다음 세션 추천 시간: **{st.session_state.last_recommended_time}분**")
    if 'last_recommendation_status' in st.session_state:
        st.info(f"📊 상태 분석: {st.session_state.last_recommendation_status}")
    
    break_placeholder = st.empty()
    
    while st.session_state.is_break_time:
        current_time = time.time()
        break_elapsed = current_time - st.session_state.break_start_time
        break_remaining = 60 - break_elapsed  # 1분 = 60초
        
        if break_remaining > 0:
            minutes = int(break_remaining // 60)
            seconds = int(break_remaining % 60)
            break_placeholder.info(f"🛌 휴식 중... 남은 시간: {minutes:02d}:{seconds:02d}")
            time.sleep(1)
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

    # 과거 세션 데이터 시각화 및 관리
    if not st.session_state.is_measuring:
        left_col, right_col = st.columns([1, 1])
        
        # 왼쪽: 추천 시간 트렌드 그래프
        with left_col:
            st.subheader("📊 Previous Session Recommendation Trend")
            
            df_hist = pd.read_sql_query("SELECT * FROM sessions", conn)
            if not df_hist.empty:
                fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
                session_numbers = range(1, len(df_hist) + 1)
                ax_hist.plot(session_numbers, df_hist["recommended_minutes"], marker='o', color='#1f77b4')
                ax_hist.set_xlabel("Session Number")
                ax_hist.set_ylabel("Recommended Time (min)")
                ax_hist.set_title("Recommendation Trend")
                ax_hist.grid(True, alpha=0.3)
                ax_hist.set_xticks(session_numbers)
                st.pyplot(fig_hist)
            else:
                st.info("No saved session data. Start your first measurement!")
        
        # 오른쪽: 세션 관리 표
        with right_col:
            st.subheader("📋 Session Management")
            
            if not df_hist.empty:
                session_table = pd.DataFrame({
                    'Session #': range(1, len(df_hist) + 1),
                    'Date': df_hist['session_date'].fillna('N/A'),
                    'Duration (min)': df_hist['session_duration'].fillna(0).round(1),
                    'Avg Attention': df_hist['attention'].round(3),
                    'Recommended (min)': df_hist['recommended_minutes'].round(1)
                })
                
                session_table = session_table.iloc[::-1].reset_index(drop=True)
                session_table['Session #'] = range(len(df_hist), 0, -1)
                
                st.markdown("""
                <style>
                .dataframe td, .dataframe th {
                    text-align: center !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.dataframe(
                    session_table, 
                    use_container_width=True,
                    hide_index=True,
                    height=250
                )
                
                # 통계 정보 표시
                st.markdown("### 📈 Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_sessions = len(df_hist)
                    st.metric("Total Sessions", total_sessions)
                
                with col2:
                    avg_attention = df_hist['attention'].mean()
                    st.metric("Avg Attention", f"{avg_attention:.3f}")
                
                with col3:
                    avg_recommended = df_hist['recommended_minutes'].mean()
                    st.metric("Avg Recommended", f"{avg_recommended:.1f}min")
                    
            else:
                st.info("No session data available yet. Complete your first session to see management data!")
        
        # Reset 버튼
        st.markdown("---")
        reset_col = st.columns([2, 1, 2])[1]
        with reset_col:
            if st.button("🗑️ Reset All Sessions"):
                cursor.execute("DELETE FROM sessions")
                conn.commit()
                st.success("Session history has been cleared.")
                st.rerun()

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

    # UI 레이아웃 (측정 중)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        emotion_placeholder = st.empty()
        timer_placeholder = st.empty()
        frame_placeholder = st.empty()
        
    with col2:
        graph_placeholder = st.empty()

    # 실시간 측정 루프
    while st.session_state.is_measuring:
        current_time = time.time()
        
        # 경과 시간 계산
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

                # 개선된 추천 로직 사용
                recommended_time, recommendation_status = calculate_recommendation_with_improved_logic(df_grouped)
                
                # 랜덤포레스트 모델 예측도 함께 사용 (하이브리드 방식)
                model_prediction = round(float(model.predict(df_grouped)[0]), 2)
                
                # 두 방식의 평균을 최종 추천으로 사용
                final_recommendation = round((recommended_time + model_prediction) / 2, 2)
                
                st.session_state.last_recommended_time = final_recommendation
                st.session_state.last_recommendation_status = recommendation_status

                # 현재 시간과 세션 시간 저장
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                session_duration = session_time

                # DB 저장
                cursor.execute("""
                    INSERT INTO sessions (angry, disgust, fear, happy, sad, surprise, neutral, attention, recommended_minutes, session_date, session_duration)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, tuple(df_grouped.iloc[0]) + (final_recommendation, current_datetime, session_duration))
                conn.commit()

                # CSV 저장 (학습 데이터 축적)
                synthetic_path = "synthetic_sessions.csv"
                df_grouped["recommended_minutes"] = final_recommendation
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

                # 타이머 표시
                remaining_time = session_time * 60 - effective_elapsed
                minutes = int(remaining_time // 60)
                seconds = int(remaining_time % 60)
                status = "⏸️ 일시정지" if st.session_state.is_paused else "▶️ 측정 중"
                timer_placeholder.markdown(f"**{status}**  \n남은 시간: `{minutes:02d}:{seconds:02d}`")

                # 영상 표시
                frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_display = cv2.flip(frame_display, 1)
                frame_placeholder.image(frame_display, channels="RGB")

                # 집중도 그래프 업데이트
                if timestamps and attn_scores:
                    fig, ax = plt.subplots()
                    fig.set_size_inches(6, 4)
                    ax.plot(timestamps, attn_scores)
                    ax.set_ylim(0, 1)
                    ax.set_title("Real-time Attention")
                    ax.set_xlabel("Time (seconds)")
                    ax.set_ylabel("Attention Score")
                    graph_placeholder.pyplot(fig, use_container_width=True)

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
=======
# 타이머 '시작,일시정지,재시작,초기화' 버튼 추가 + 남은시간 표시, 초기화 버튼 누르면 수집된 데이터는 저장되지 않고 삭제
import streamlit as st                       # Streamlit 웹 앱 UI 생성을 위한 라이브러리
import cv2                                   # OpenCV: 영상 처리용 라이브러리 (카메라 입력 등)
import numpy as np                           # Numpy: 수치 계산용 라이브러리
from fer import FER                          # FER: 감정 인식 라이브러리
import mediapipe as mp                       # MediaPipe: 얼굴 랜드마크 추출 라이브러리
import pandas as pd                          # Pandas: 데이터프레임 처리를 위한 라이브러리
import time                                  # 시간 측정용 라이브러리
import matplotlib.pyplot as plt              # 시각화를 위한 라이브러리
from sklearn.ensemble import RandomForestRegressor  # 랜덤 포레스트 회귀 모델
import sqlite3                               # SQLite 데이터베이스 연동용 라이브러리
import os                                    # OS 관련 함수 (파일 존재 확인 등)

# Streamlit 페이지 레이아웃 설정
st.set_page_config(layout="wide")

# 페이지 제목 설정
st.title("🎯 감정/집중도 기반 맞춤형 뽀모도로 타이머")

# --- 세션 상태 초기화 ---
if 'is_measuring' not in st.session_state:
    st.session_state.is_measuring = False
if 'is_paused' not in st.session_state:
    st.session_state.is_paused = False
if 'total_elapsed_time' not in st.session_state:
    st.session_state.total_elapsed_time = 0
if 'pause_start_time' not in st.session_state:
    st.session_state.pause_start_time = 0

# --- 데이터베이스 초기화 ---
conn = sqlite3.connect("sessions.db", check_same_thread=False)  # SQLite DB 연결
cursor = conn.cursor()                                          # 커서 생성
cursor.execute('''                                              
CREATE TABLE IF NOT EXISTS sessions (                           
    id INTEGER PRIMARY KEY AUTOINCREMENT,                      
    angry REAL, disgust REAL, fear REAL, happy REAL, sad REAL,  
    surprise REAL, neutral REAL,                                
    attention REAL,                                             
    recommended_minutes REAL                                   
)
''')
conn.commit()  # DB에 변경사항 저장

# --- 사용자 UI: 세션 시간 입력 ---
st.markdown("## ⌚ 세션 시간 설정")
session_time = st.number_input("측정할 세션 시간 (분)", min_value=0.5, max_value=60.0, value=25.0, step=0.5)
# 사용자가 측정 시간을 설정할 수 있는 입력창

# --- 과거 세션 데이터 시각화 ---
if session_time == 25.0:  # 기본값일 때만 시각화 표시
    st.subheader("📊 Previous Session Recommendation Trend")

    col_a, col_b = st.columns([4, 1])  # 시각화와 리셋 버튼을 좌우 컬럼으로 분리
    with col_a:
        df_hist = pd.read_sql_query("SELECT * FROM sessions", conn)  # 과거 세션 데이터 불러오기
        if not df_hist.empty:  # 데이터가 있을 경우
            fig_hist, ax_hist = plt.subplots()
            ax_hist.plot(df_hist.index + 1, df_hist["recommended_minutes"], marker='o')  # 세션별 추천 시간 그래프
            ax_hist.set_xlabel("Session Number")
            ax_hist.set_ylabel("Recommended Time (min)")
            ax_hist.set_title("Recommendation Trend")
            st.pyplot(fig_hist)  # Streamlit에 그래프 표시
        else:
            st.info("No saved session data. Start your first measurement!")  # 데이터 없을 경우 안내 메시지
    with col_b:
        if st.button("🗑️ Reset Sessions"):  # 데이터 초기화 버튼
            cursor.execute("DELETE FROM sessions")
            conn.commit()
            st.success("Session history has been cleared.")
            st.rerun()  # 페이지 새로고침

# --- 모델 학습 함수 정의 ---
@st.cache_resource  # Streamlit 캐시: 학습은 한 번만 수행되도록 함
def train_model():
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
    
    df = pd.read_csv(synthetic_path)  # 과거 학습용 CSV 불러오기
    X = df[df.columns[:-1]]                     # 입력 데이터 (감정 + 집중도)
    y = df["recommended_minutes"]               # 타깃 값 (추천 시간)
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # 랜덤포레스트 회귀 모델 정의
    model.fit(X, y)                             # 모델 학습
    return model

model = train_model()  # 학습된 모델 가져오기

# --- 제어 버튼들 ---
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
            # 일시정지된 시간을 총 경과 시간에 추가
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

# --- 측정 시작 ---
if st.session_state.is_measuring:
    emotion_detector = FER(mtcnn=False)                       # 감정 인식기 초기화
    mp_face_mesh = mp.solutions.face_mesh                    # MediaPipe 얼굴 랜드마크 솔루션
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)  # 실시간 얼굴 추적기
    
    # 세션 상태에 카메라와 데이터 저장
    if 'cap' not in st.session_state or st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)                # 카메라 켜기 (0번 기본 웹캠)
        st.session_state.start_time = time.time()                 # 측정 시작 시간 기록
        st.session_state.data = []                                # 결과 저장용 리스트들 초기화
        st.session_state.timestamps = []
        st.session_state.attn_scores = []

    cap = st.session_state.cap
    start_time = st.session_state.start_time
    data = st.session_state.data
    timestamps = st.session_state.timestamps
    attn_scores = st.session_state.attn_scores

    # Streamlit 컬럼 나누기: 영상/그래프 vs. 텍스트 출력
    col1, col2 = st.columns([2, 1])
    with col1:
        frame_placeholder = st.empty()       # 영상 프레임 표시용 공간
        graph_placeholder = st.empty()       # 그래프 표시용 공간
    with col2:
        emotion_placeholder = st.empty()     # 감정 상태 표시용 공간
        timer_placeholder = st.empty()       # 타이머 표시용 공간
        result_box = st.empty()              # 최종 결과 출력용 공간

    # --- 실시간 카메라 루프 시작 ---
    while st.session_state.is_measuring:
        current_time = time.time()
        
        # 일시정지 상태가 아닐 때만 시간 계산
        if not st.session_state.is_paused:
            effective_elapsed = (current_time - start_time) - st.session_state.total_elapsed_time
        else:
            # 일시정지 중이면 마지막 측정 시간 유지
            if timestamps:
                effective_elapsed = timestamps[-1]
            else:
                effective_elapsed = 0

        # 세션 시간 체크
        if effective_elapsed >= session_time * 60:
            # 측정 완료
            cap.release()          # 카메라 종료
            cv2.destroyAllWindows()  # OpenCV 창 닫기
            st.session_state.cap = None
            st.session_state.is_measuring = False
            st.session_state.is_paused = False

            if data:  # 데이터가 있을 경우에만 처리
                df = pd.DataFrame(data)                                # 전체 데이터프레임 변환
                avg_attention = df["attention"].mean()                 # 평균 집중도 계산
                df_grouped = df[['angry','disgust','fear','happy','sad','surprise','neutral','attention']].mean().to_frame().T  # 평균값으로 1줄 만들기

                # 모델 예측 수행
                recommended_time = round(float(model.predict(df_grouped)[0]), 2)  # 추천 시간 예측
                result_box.success(f"✅ 측정 완료! 추천 시간: **{recommended_time}분**")  # 결과 출력

                # DB에 저장
                cursor.execute("""
                    INSERT INTO sessions (angry, disgust, fear, happy, sad, surprise, neutral, attention, recommended_minutes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, tuple(df_grouped.iloc[0]) + (recommended_time,))
                conn.commit()

                # CSV 파일에도 저장 (학습에 활용됨)
                synthetic_path = "synthetic_sessions.csv"
                df_grouped["recommended_minutes"] = recommended_time
                if os.path.exists(synthetic_path):  # 기존 파일이 있으면 이어쓰기
                    existing = pd.read_csv(synthetic_path)
                    updated = pd.concat([existing, df_grouped], ignore_index=True)
                    updated.to_csv(synthetic_path, index=False)
                else:
                    df_grouped.to_csv(synthetic_path, index=False)  # 없으면 새로 생성
            
            # 상태 초기화
            st.session_state.total_elapsed_time = 0
            st.session_state.pause_start_time = 0
            break
        
        else:
            # 측정 진행 중
            ret, frame = cap.read()  # 프레임 읽기
            if ret:
                # 일시정지 상태가 아닐 때만 데이터 수집
                if not st.session_state.is_paused:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환
                    h, w, _ = frame.shape                             # 이미지 크기
                    results = face_mesh.process(img_rgb)              # 얼굴 랜드마크 검출

                    emotions = emotion_detector.detect_emotions(frame)  # 감정 인식 수행
                    if emotions:
                        top = emotions[0]                           # 첫 번째 얼굴 정보 사용
                        (x, y, w_box, h_box) = top["box"]           # 얼굴 박스 좌표
                        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)  # 얼굴 표시
                        emotion_text = ", ".join([f"{k}: {v:.2f}" for k, v in top["emotions"].items()])  # 감정 텍스트
                        emotion_dict = top["emotions"]              # 감정 사전
                    else:
                        emotion_text = "얼굴 감지 안됨"             # 얼굴이 없을 경우
                        emotion_dict = {k: 0 for k in ['angry','disgust','fear','happy','sad','surprise','neutral']}  # 기본값

                    attention = 0
                    if results.multi_face_landmarks:  # 얼굴이 감지된 경우
                        landmarks = results.multi_face_landmarks[0]
                        points = [(lm.x * w, lm.y * h) for lm in landmarks.landmark]  # 얼굴 좌표 계산
                        left_eye = np.mean([points[33], points[133]], axis=0)         # 왼쪽 눈 중심
                        right_eye = np.mean([points[362], points[263]], axis=0)       # 오른쪽 눈 중심
                        eye_center = (left_eye + right_eye) / 2                       # 두 눈 중간 지점
                        screen_center = np.array([w / 2, h / 2])                      # 화면 중심
                        dist = np.linalg.norm(eye_center - screen_center)            # 중심과의 거리
                        attention = max(0, 1 - dist / (w / 2))                        # 거리로부터 집중도 계산 (1에 가까울수록 화면 중심)

                    timestamps.append(effective_elapsed)            # 타임스탬프 저장
                    attn_scores.append(attention)                   # 집중도 저장

                    data.append({                                   # 한 프레임의 감정 + 집중도 저장
                        'timestamp': effective_elapsed,
                        **emotion_dict,
                        'attention': attention
                    })

                    # 상태 업데이트
                    st.session_state.data = data
                    st.session_state.timestamps = timestamps
                    st.session_state.attn_scores = attn_scores

                    emotion_placeholder.markdown(f"**감정 상태**: {emotion_text}\n**집중도**: `{attention:.2f}`")  # 텍스트 출력
                else:
                    # 일시정지 중일 때는 마지막 감정 상태 유지
                    emotion_placeholder.markdown("⏸️ **일시정지 중**")

                # 영상은 일시정지 중에도 계속 표시
                frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # 영상 변환 후 표시
                frame_placeholder.image(frame_display, channels="RGB")     # 실시간 영상 출력

                # 그래프 업데이트 (데이터가 있을 때만)
                if timestamps and attn_scores:
                    fig, ax = plt.subplots()
                    fig.set_size_inches(5, 2.5)
                    ax.plot(timestamps, attn_scores)                           # 집중도 실시간 그래프
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
                st.error("카메라 오류")  # 오류 시 종료
                st.session_state.is_measuring = False
                if st.session_state.cap:
                    st.session_state.cap.release()
                    st.session_state.cap = None
                break

# 카메라 정리 (세션 종료 시)
if not st.session_state.is_measuring and 'cap' in st.session_state and st.session_state.cap is not None:
    st.session_state.cap.release()
    st.session_state.cap = None
