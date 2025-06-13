#dfsfd
# === 1. 라이브러리 import ===
import streamlit as st                       # Streamlit: 웹 앱 UI를 쉽게 만들 수 있는 Python 라이브러리
import cv2                                   # OpenCV: 컴퓨터 비전, 영상 처리용 라이브러리 (카메라 캡처, 이미지 처리)
import numpy as np                           # Numpy: 수치 계산, 배열 연산을 위한 기본 라이브러리
from fer import FER                          # FER: 얼굴 감정 인식(Facial Emotion Recognition) 라이브러리
import mediapipe as mp                       # MediaPipe: 구글의 머신러닝 프레임워크 (얼굴 랜드마크 추출용)
import pandas as pd                          # Pandas: 데이터프레임 처리, 데이터 조작용 라이브러리
import time                                  # time: 시간 측정, 지연 등 시간 관련 함수 제공
import matplotlib.pyplot as plt              # matplotlib: 그래프, 차트 등 데이터 시각화 라이브러리
from sklearn.ensemble import RandomForestRegressor  # scikit-learn의 랜덤 포레스트 회귀 모델 (머신러닝)
import sqlite3                               # SQLite: 경량 데이터베이스 연동 라이브러리
import os                                    # os: 운영체제 관련 함수 (파일 경로, 프로세스 관리 등)
import winsound                              # winsound: Windows 시스템 소리 재생용 (알람 소리)
from datetime import datetime                # datetime: 날짜와 시간 처리용 라이브러리
import signal                                # signal: 프로그램 신호 처리 (프로그램 종료 등)

# === 2. Streamlit 페이지 설정 ===
st.set_page_config(layout="wide")            # Streamlit 페이지를 와이드 레이아웃으로 설정 (화면을 넓게 사용)

# 제목과 종료 버튼을 한 줄에 배치
title_col, exit_col = st.columns([5, 1])    # 화면을 5:1 비율로 두 개 컬럼으로 나눔 (제목용:버튼용)
with title_col:                              # 첫 번째 컬럼 (넓은 부분)에서
    st.title("🎯 감정/집중도 기반 맞춤형 Pomodoro Timer")  # 앱의 메인 제목 표시
with exit_col:                               # 두 번째 컬럼 (좁은 부분)에서
    if st.button("🚪 타이머 중지(종료)", type="secondary"):  # 종료 버튼 생성, 클릭 시 아래 코드 실행
        # 측정 중이면 카메라 해제
        if 'cap' in st.session_state and st.session_state.cap is not None:  # 카메라가 활성화되어 있다면
            st.session_state.cap.release()  # 카메라 리소스 해제
            cv2.destroyAllWindows()         # OpenCV 윈도우 모두 닫기
        
        st.error("🚪 프로그램 종료 중...")    # 에러 메시지로 종료 중임을 표시
        st.info("터미널에서 Ctrl+C를 눌러 완전히 종료하거나, 브라우저 탭을 닫아주세요.")  # 사용자 안내 메시지
        
        # 프로세스 종료 시도
        try:
            os.kill(os.getpid(), signal.SIGTERM)  # 현재 프로세스 ID를 얻어서 SIGTERM 신호로 종료 시도
        except:                              # 종료 실패 시
            st.stop()                       # Streamlit의 실행 중지 함수 호출

# === 3. 세션 상태 초기화 ===
# Streamlit의 session_state는 페이지 새로고침 간에도 데이터를 유지하는 저장소
if 'is_measuring' not in st.session_state:   # 'is_measuring' 키가 세션 상태에 없다면
    st.session_state.is_measuring = False    # 측정 중 상태를 False로 초기화
if 'is_paused' not in st.session_state:     # 일시정지 상태 확인
    st.session_state.is_paused = False      # 일시정지 상태를 False로 초기화
if 'total_elapsed_time' not in st.session_state:  # 총 경과 시간 확인
    st.session_state.total_elapsed_time = 0 # 총 경과 시간을 0으로 초기화
if 'pause_start_time' not in st.session_state:    # 일시정지 시작 시간 확인
    st.session_state.pause_start_time = 0   # 일시정지 시작 시간을 0으로 초기화
if 'is_break_time' not in st.session_state: # 휴식 시간 상태 확인
    st.session_state.is_break_time = False  # 휴식 시간 상태를 False로 초기화
if 'break_start_time' not in st.session_state:    # 휴식 시작 시간 확인
    st.session_state.break_start_time = 0   # 휴식 시작 시간을 0으로 초기화
if 'recommended_time' not in st.session_state:    # 추천 시간 확인
    st.session_state.recommended_time = 25.0       # 기본 추천 시간을 25분으로 설정
if 'show_alarm' not in st.session_state:   # 알람 표시 상태 확인
    st.session_state.show_alarm = False     # 알람 표시 상태를 False로 초기화

# === 4. 데이터베이스 초기화 ===
conn = sqlite3.connect("sessions.db", check_same_thread=False)  # SQLite 데이터베이스 연결 (멀티스레드 허용)
cursor = conn.cursor()                      # 데이터베이스 커서 생성 (쿼리 실행용)
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
conn.commit()                               # 데이터베이스 변경사항 저장

# === 5. 함수 정의 ===
def play_alarm():
    """알람 소리 재생 함수"""
    try:
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)  # Windows 시스템 경고음 재생
        for _ in range(3):                  # 3번 반복
            winsound.Beep(1000, 500)       # 1000Hz 주파수로 0.5초간 비프음 재생
            time.sleep(0.2)                # 0.2초 대기
    except:                                 # 소리 재생 실패 시 (비Windows 환경 등)
        print("🔔 알람! 시간이 완료되었습니다!")  # 콘솔에 텍스트 알람 출력

def generate_synthetic_data():
    """개선된 가상 데이터 생성 함수 (data.py 로직 적용)"""
    np.random.seed(42)                      # 랜덤 시드 고정 (재현 가능한 결과)
    n = 200                                 # 생성할 데이터 개수
    
    # 베타분포를 사용한 현실적인 감정 분포
    df = pd.DataFrame({                     # 판다스 데이터프레임 생성
        "angry": np.random.beta(3, 4, n),            # 화남: 베타분포(3,4) - 중간~높은 값 분포
        "disgust": np.random.beta(1, 9, n),          # 혐오: 베타분포(1,9) - 거의 0에 가까운 값
        "fear": np.random.beta(3, 4, n),             # 두려움: 베타분포(3,4) - 중간~높은 값 분포
        "happy": np.random.beta(1, 9, n),            # 행복: 베타분포(1,9) - 거의 0에 가까운 값
        "sad": np.random.beta(3, 4, n),              # 슬픔: 베타분포(3,4) - 중간~높은 값 분포
        "surprise": np.random.beta(1, 9, n),         # 놀람: 베타분포(1,9) - 거의 0에 가까운 값
        "neutral": np.random.beta(4, 3, n),          # 중립: 베타분포(4,3) - 높은 값 분포
        "attention": np.random.uniform(0.3, 0.95, n) # 집중도: 0.3~0.95 균등분포
    })
    
    # 개선된 가중치 (data.py에서 가져온 값)
    weights = np.array([                    # 각 감정별 가중치 배열
        1.6,    # angry: 화남이 높으면 추천 시간 증가
        0.0,    # disgust: 혐오는 의미 없음 (가중치 0)
        1.6,    # fear: 두려움이 높으면 추천 시간 증가
        0.0,    # happy: 행복은 의미 없음 (가중치 0)
        1.6,    # sad: 슬픔이 높으면 추천 시간 증가
        0.0,    # surprise: 놀람은 의미 없음 (가중치 0)
        -0.5,   # neutral: 중립이 높으면 추천 시간 감소 (더 높은 집중 유도)
        0.8     # attention: 집중도가 높으면 추천 시간 증가
    ])
    
    # 노이즈 추가
    noise = np.random.normal(0, 1.5, n)    # 평균 0, 표준편차 1.5인 정규분포 노이즈
    
    # 추천 시간 계산
    df["recommended_minutes"] = 25 + df[df.columns].values @ weights + noise  # 기본 25분 + 가중합 + 노이즈
    
    # 범위 조정 (20-50분)
    df["recommended_minutes"] = np.clip(df["recommended_minutes"], 20, 50)     # 20~50분 범위로 제한
    
    # 특별한 경우 처리: attention이 매우 낮으면 30분 이상
    low_attention_mask = df["attention"] < 0.55                               # 집중도가 0.55 미만인 경우
    df.loc[low_attention_mask, "recommended_minutes"] = np.clip(              # 해당 케이스들의 추천 시간을
        df.loc[low_attention_mask, "recommended_minutes"], 30, 50             # 30~50분 범위로 재조정
    )
    
    return df                               # 생성된 데이터프레임 반환

def calculate_recommendation_with_improved_logic(df_grouped):
    """개선된 추천 로직 (session_core.py 로직 적용 - 집중 감정 3개 모두 고려)"""
    # 집중 상태 분석 - angry, fear, sad 모두 고려
    angry_score = df_grouped['angry'].iloc[0]       # 화남 점수 추출
    fear_score = df_grouped['fear'].iloc[0]         # 두려움 점수 추출
    sad_score = df_grouped['sad'].iloc[0]           # 슬픔 점수 추출
    neutral_score = df_grouped['neutral'].iloc[0]   # 중립 점수 추출
    attention_score = df_grouped['attention'].iloc[0]  # 집중도 점수 추출
    
    # 집중 감정 점수 합계 계산 (session_core.py와 동일)
    concentration_emotion_score = angry_score + fear_score + sad_score  # 집중 관련 감정들의 합
    
    # 집중 상태별 시간 추천 로직 (수정된 부분)
    if concentration_emotion_score > 0.25 and attention_score >= 0.55:  # 매우 집중 상태 조건
        recommended_time = 25 + concentration_emotion_score * 15 + attention_score * 10  # 기본시간 + 보너스
        status = f"매우 집중 (Angry: {angry_score:.2f}, Fear: {fear_score:.2f}, Sad: {sad_score:.2f}, 집중감정합계: {concentration_emotion_score:.2f}, Attention: {attention_score:.2f}) → 집중력 유지하며 시간 증가"
    elif neutral_score >= 0.55 and attention_score >= 0.55: # 보통 집중 상태 조건
        recommended_time = max(20, 25 - neutral_score * 8)  # 기본시간에서 중립도에 따라 감소, 최소 20분
        status = f"보통 집중 (Neutral: {neutral_score:.2f}, 집중감정합계: {concentration_emotion_score:.2f}, Attention: {attention_score:.2f}) → 더 높은 집중 유도를 위해 시간 단축"
    else:  # 산만한 상태
        recommended_time = 30.0             # 산만할 때는 30분 고정
        status = f"산만함 (Neutral: {neutral_score:.2f}, 집중감정합계: {concentration_emotion_score:.2f}, Attention: {attention_score:.2f}) → 차분히 앉아있기 위해 긴 시간 권장"
    
    # 추천 시간 범위 제한 (20-50분)
    recommended_time = np.clip(recommended_time, 20, 50)  # 최종 추천 시간을 20~50분으로 제한
    
    return recommended_time, status         # 추천 시간과 상태 설명 반환

@st.cache_resource                          # Streamlit 캐시 데코레이터 (리소스 캐싱)
def train_model():
    """머신러닝 모델 학습 함수 (개선된 데이터 사용)"""
    synthetic_path = "synthetic_sessions.csv"       # 학습 데이터 파일 경로
    if not os.path.exists(synthetic_path):          # 파일이 존재하지 않으면
        # 개선된 학습 데이터 생성
        df = generate_synthetic_data()              # 가상 데이터 생성
        df.to_csv(synthetic_path, index=False)      # CSV 파일로 저장 (인덱스 제외)
        st.info("🔄 개선된 학습 데이터가 생성되었습니다.")  # 사용자에게 알림
    
    df = pd.read_csv(synthetic_path)                # CSV 파일에서 데이터 읽기
    X = df[df.columns[:-1]]                         # 마지막 컬럼 제외한 모든 컬럼 (특성)
    y = df["recommended_minutes"]                   # 마지막 컬럼 (타겟 변수)
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # 랜덤포레스트 회귀 모델 생성
    model.fit(X, y)                                 # 모델 학습
    return model                                    # 학습된 모델 반환

# 모델 학습
model = train_model()                       # 모델 학습 함수 호출하여 모델 생성

# === 6. 메인 UI 분기 ===

# --- 6-1. 휴식 시간 처리 ---
if st.session_state.is_break_time:          # 휴식 시간 상태라면
    st.markdown("## ☕ 휴식 시간")          # 휴식 시간 제목 표시
    
    # 추천 시간과 상태 표시
    if 'last_recommended_time' in st.session_state:           # 마지막 추천 시간이 저장되어 있다면
        st.success(f"✅ 다음 세션 추천 시간: **{st.session_state.last_recommended_time}분**")  # 성공 메시지로 표시
    if 'last_recommendation_status' in st.session_state:      # 마지막 추천 상태가 저장되어 있다면
        st.info(f"📊 상태 분석: {st.session_state.last_recommendation_status}")  # 정보 메시지로 표시
    
    break_placeholder = st.empty()          # 빈 컨테이너 생성 (동적 업데이트용)
    
    while st.session_state.is_break_time:   # 휴식 시간 동안 반복
        current_time = time.time()          # 현재 시간 가져오기
        break_elapsed = current_time - st.session_state.break_start_time  # 휴식 경과 시간 계산
        break_remaining = 300 - break_elapsed  # 남은 휴식 시간 (5분 = 300초)
        
        if break_remaining > 0:             # 휴식 시간이 남아있다면
            minutes = int(break_remaining // 60)    # 남은 분 계산
            seconds = int(break_remaining % 60)     # 남은 초 계산
            break_placeholder.info(f"🛌 휴식 중... 남은 시간: {minutes:02d}:{seconds:02d}")  # 남은 시간 표시
            time.sleep(1)                   # 1초 대기
        else:                               # 휴식 시간이 끝났다면
            # 휴식 시간 완료
            st.session_state.is_break_time = False      # 휴식 시간 상태 해제
            st.session_state.break_start_time = 0       # 휴식 시작 시간 초기화
            if 'last_recommended_time' in st.session_state:  # 마지막 추천 시간이 있다면
                st.session_state.recommended_time = st.session_state.last_recommended_time  # 추천 시간 업데이트
            play_alarm()                    # 알람 재생
            break_placeholder.success("🎉 휴식 시간이 완료되었습니다!")  # 완료 메시지 표시
            st.rerun()                      # 페이지 새로고침
            break                           # while 루프 종료

# --- 6-2. 일반 모드 (측정 준비/진행 중) ---
else:                                       # 휴식 시간이 아니라면
    # 세션 시간 설정
    st.markdown("## ⌚ 세션 시간 설정")    # 세션 시간 설정 제목
    session_time = st.number_input(         # 숫자 입력 위젯 생성
        "측정할 세션 시간(분)",             # 라벨
        min_value=0.5,                      # 최소값 0.5분
        max_value=60.0,                     # 최대값 60분
        value=st.session_state.recommended_time,  # 기본값 (추천 시간)
        step=0.5                            # 증감 단위 0.5분
    )

    # 제어 버튼들
    st.markdown("## 🎮 세션 제어")        # 세션 제어 제목
    button_col1, button_col2, button_col3 = st.columns(3)  # 3개 컬럼으로 버튼 배치

    with button_col1:                       # 첫 번째 컬럼
        if st.button("▶ 측정 시작", disabled=st.session_state.is_measuring):  # 측정 시작 버튼 (측정 중이면 비활성화)
            st.session_state.is_measuring = True       # 측정 상태 활성화
            st.session_state.is_paused = False         # 일시정지 상태 해제
            st.session_state.total_elapsed_time = 0    # 총 경과 시간 초기화
            st.session_state.pause_start_time = 0      # 일시정지 시작 시간 초기화

    with button_col2:                       # 두 번째 컬럼
        if st.button("⏸️ 일시정지", disabled=not st.session_state.is_measuring or st.session_state.is_paused):  # 일시정지 버튼
            st.session_state.is_paused = True          # 일시정지 상태 활성화
            st.session_state.pause_start_time = time.time()  # 일시정지 시작 시간 기록

    with button_col3:                       # 세 번째 컬럼
        if st.button("▶️ 재시작", disabled=not st.session_state.is_measuring or not st.session_state.is_paused):  # 재시작 버튼
            if st.session_state.is_paused:             # 일시정지 상태라면
                pause_duration = time.time() - st.session_state.pause_start_time  # 일시정지 지속 시간 계산
                st.session_state.total_elapsed_time += pause_duration  # 총 경과 시간에 일시정지 시간 추가
                st.session_state.is_paused = False     # 일시정지 상태 해제
                st.session_state.pause_start_time = 0  # 일시정지 시작 시간 초기화

    # 초기화 버튼
    if st.session_state.is_measuring:       # 측정 중이라면
        if st.button("⏹️ 초기화"):          # 초기화 버튼
            st.session_state.is_measuring = False      # 측정 상태 해제
            st.session_state.is_paused = False         # 일시정지 상태 해제
            st.session_state.total_elapsed_time = 0    # 총 경과 시간 초기화
            st.session_state.pause_start_time = 0      # 일시정지 시작 시간 초기화
            st.info("초기화되었습니다.")               # 초기화 완료 메시지

    # 과거 세션 데이터 시각화 및 관리
    if not st.session_state.is_measuring:   # 측정 중이 아니라면
        left_col, right_col = st.columns([1, 1])  # 1:1 비율로 두 컬럼 생성
        
        # 왼쪽: 추천 시간 트렌드 그래프
        with left_col:                      # 왼쪽 컬럼
            st.subheader("📊 Previous Session Recommendation Trend")  # 그래프 제목
            
            df_hist = pd.read_sql_query("SELECT * FROM sessions", conn)  # DB에서 모든 세션 데이터 조회
            if not df_hist.empty:           # 데이터가 있다면
                fig_hist, ax_hist = plt.subplots(figsize=(6, 4))  # matplotlib 그래프 생성
                session_numbers = range(1, len(df_hist) + 1)      # 세션 번호 생성 (1부터 시작)
                ax_hist.plot(session_numbers, df_hist["recommended_minutes"], marker='o', color='#1f77b4')  # 선 그래프 그리기
                ax_hist.set_xlabel("Session Number")              # x축 라벨
                ax_hist.set_ylabel("Recommended Time (min)")      # y축 라벨
                ax_hist.set_title("Recommendation Trend")         # 그래프 제목
                ax_hist.grid(True, alpha=0.3)                     # 격자 표시 (투명도 0.3)
                ax_hist.set_xticks(session_numbers)               # x축 눈금 설정
                st.pyplot(fig_hist)                               # Streamlit에 그래프 표시
            else:                           # 데이터가 없다면
                st.info("No saved session data. Start your first measurement!")  # 안내 메시지
        
        # 오른쪽: 세션 관리 표
        with right_col:                     # 오른쪽 컬럼
            st.subheader("📋 Session Management")  # 테이블 제목
            
            if not df_hist.empty:           # 데이터가 있다면
                session_table = pd.DataFrame({      # 표시용 데이터프레임 생성
                    'Session #': range(1, len(df_hist) + 1),        # 세션 번호
                    'Date': df_hist['session_date'].fillna('N/A'),  # 날짜 (없으면 N/A)
                    'Duration (min)': df_hist['session_duration'].fillna(0).round(1),  # 세션 지속시간
                    'Avg Attention': df_hist['attention'].round(3), # 평균 집중도
                    'Recommended (min)': df_hist['recommended_minutes'].round(1)  # 추천 시간
                })
                
                session_table = session_table.iloc[::-1].reset_index(drop=True)  # 순서 뒤집기 (최신 데이터가 위로)
                session_table['Session #'] = range(len(df_hist), 0, -1)         # 세션 번호 재정렬
                
                st.markdown("""                    # CSS 스타일 적용
                <style>
                .dataframe td, .dataframe th {
                    text-align: center !important;     # 테이블 내용 가운데 정렬
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.dataframe(                       # 데이터프레임을 테이블로 표시
                    session_table,                  # 표시할 데이터
                    use_container_width=True,       # 컨테이너 너비에 맞춤
                    hide_index=True,                # 인덱스 숨김
                    height=250                      # 높이 250px
                )
                
                # 통계 정보 표시
                st.markdown("### 📈 Statistics")   # 통계 제목
                col1, col2, col3 = st.columns(3)   # 3개 컬럼으로 통계 표시
                
                with col1:                          # 첫 번째 컬럼에서
                    total_sessions = len(df_hist)   # 전체 세션 개수 계산
                    st.metric("Total Sessions", total_sessions)  # 메트릭 위젯으로 총 세션 수 표시
                
                with col2:                          # 두 번째 컬럼에서
                    avg_attention = df_hist['attention'].mean()  # 모든 세션의 평균 집중도 계산
                    st.metric("Avg Attention", f"{avg_attention:.3f}")  # 평균 집중도를 소수점 3자리까지 표시
                
                with col3:                          # 세 번째 컬럼에서
                    avg_recommended = df_hist['recommended_minutes'].mean()  # 모든 세션의 평균 추천 시간 계산
                    st.metric("Avg Recommended", f"{avg_recommended:.1f}min")  # 평균 추천 시간을 소수점 1자리까지 표시
                    
            else:                                   # 세션 데이터가 없는 경우
                st.info("No session data available yet. Complete your first session to see management data!")  # 안내 메시지 표시
        
        # Reset 버튼 섹션
        st.markdown("---")                          # 구분선 추가 (마크다운 형식)
        reset_col = st.columns([2, 1, 2])[1]       # 2:1:2 비율로 컬럼을 나누고 가운데 컬럼만 선택 (버튼을 중앙에 배치)
        with reset_col:                             # 가운데 컬럼에서
            if st.button("🗑️ Reset All Sessions"):  # 모든 세션 삭제 버튼 생성, 클릭 시 아래 코드 실행
                cursor.execute("DELETE FROM sessions")  # SQL: sessions 테이블의 모든 데이터 삭제
                conn.commit()                       # 데이터베이스 변경사항 저장
                st.success("Session history has been cleared.")  # 성공 메시지 표시
                st.rerun()                          # 페이지 새로고침으로 화면 업데이트

# === 7. 실시간 측정 처리 ===
if st.session_state.is_measuring:              # 측정 중인 상태라면
    # MediaPipe 및 FER 초기화
    emotion_detector = FER(mtcnn=False)         # FER 감정 인식 객체 생성 (MTCNN 얼굴 검출 비활성화로 속도 향상)
    mp_face_mesh = mp.solutions.face_mesh       # MediaPipe 얼굴 메쉬 솔루션 가져오기
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)  # 얼굴 메쉬 객체 생성 (동영상 모드, 최대 1개 얼굴)
    
    # 카메라 및 데이터 초기화
    if 'cap' not in st.session_state or st.session_state.cap is None:  # 카메라 객체가 세션에 없거나 None이면
        st.session_state.cap = cv2.VideoCapture(0)  # 첫 번째 카메라(인덱스 0) 캡처 객체 생성
        st.session_state.start_time = time.time()   # 측정 시작 시간 기록
        st.session_state.data = []                  # 측정 데이터를 저장할 빈 리스트 초기화
        st.session_state.timestamps = []            # 타임스탬프를 저장할 빈 리스트 초기화
        st.session_state.attn_scores = []           # 집중도 점수를 저장할 빈 리스트 초기화

    cap = st.session_state.cap                      # 세션에서 카메라 객체 가져오기
    start_time = st.session_state.start_time        # 세션에서 시작 시간 가져오기
    data = st.session_state.data                    # 세션에서 데이터 리스트 가져오기
    timestamps = st.session_state.timestamps        # 세션에서 타임스탬프 리스트 가져오기
    attn_scores = st.session_state.attn_scores      # 세션에서 집중도 점수 리스트 가져오기

    # UI 레이아웃 (측정 중)
    col1, col2 = st.columns([1, 1])                # 1:1 비율로 두 개 컬럼 생성
    
    with col1:                                      # 첫 번째 컬럼 (왼쪽)에서
        emotion_placeholder = st.empty()            # 감정 정보 표시용 빈 컨테이너 생성
        timer_placeholder = st.empty()              # 타이머 표시용 빈 컨테이너 생성
        frame_placeholder = st.empty()              # 카메라 영상 표시용 빈 컨테이너 생성
        
    with col2:                                      # 두 번째 컬럼 (오른쪽)에서
        graph_placeholder = st.empty()              # 실시간 그래프 표시용 빈 컨테이너 생성

    # 실시간 측정 루프
    while st.session_state.is_measuring:           # 측정 중인 동안 계속 반복
        current_time = time.time()                 # 현재 시간 가져오기
        
        # 경과 시간 계산
        if not st.session_state.is_paused:         # 일시정지 상태가 아니라면
            effective_elapsed = (current_time - start_time) - st.session_state.total_elapsed_time  # 실제 경과 시간 = 전체 시간 - 일시정지된 시간
        else:                                      # 일시정지 상태라면
            if timestamps:                         # 타임스탬프가 있다면
                effective_elapsed = timestamps[-1]  # 마지막 타임스탬프를 경과 시간으로 사용
            else:                                  # 타임스탬프가 없다면
                effective_elapsed = 0              # 경과 시간을 0으로 설정

        # 세션 시간 완료 체크
        if effective_elapsed >= session_time * 60:  # 경과 시간이 설정된 세션 시간(분을 초로 변환)을 초과했다면
            # 측정 완료 처리
            cap.release()                          # 카메라 리소스 해제
            cv2.destroyAllWindows()                # 모든 OpenCV 윈도우 닫기
            st.session_state.cap = None            # 세션에서 카메라 객체 제거
            st.session_state.is_measuring = False  # 측정 상태 해제
            st.session_state.is_paused = False     # 일시정지 상태 해제

            # 알람 재생
            play_alarm()                           # 측정 완료 알람 소리 재생

            if data:                               # 측정된 데이터가 있다면
                # 데이터 분석 및 예측
                df = pd.DataFrame(data)            # 측정 데이터를 판다스 데이터프레임으로 변환
                avg_attention = df["attention"].mean()  # 평균 집중도 계산
                df_grouped = df[['angry','disgust','fear','happy','sad','surprise','neutral','attention']].mean().to_frame().T  # 각 감정과 집중도의 평균을 계산하여 1행 데이터프레임으로 변환

                # 개선된 추천 로직 사용 (집중 감정 3개 모두 고려)
                recommended_time, recommendation_status = calculate_recommendation_with_improved_logic(df_grouped)  # 커스텀 추천 로직으로 시간과 상태 계산
                
                # 랜덤포레스트 모델 예측도 함께 사용 (하이브리드 방식)
                model_prediction = round(float(model.predict(df_grouped)[0]), 2)  # 머신러닝 모델로 추천 시간 예측 (소수점 2자리)
                
                # 두 방식의 평균을 최종 추천으로 사용
                final_recommendation = round((recommended_time + model_prediction) / 2, 2)  # 커스텀 로직과 ML 모델의 평균을 최종 추천 시간으로 계산
                
                st.session_state.last_recommended_time = final_recommendation      # 세션에 마지막 추천 시간 저장
                st.session_state.last_recommendation_status = recommendation_status  # 세션에 마지막 추천 상태 저장

                # 현재 시간과 세션 시간 저장
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")   # 현재 날짜와 시간을 문자열로 포맷팅
                session_duration = session_time                                   # 실제 세션 지속 시간

                # DB 저장
                cursor.execute("""                                                
                    INSERT INTO sessions (angry, disgust, fear, happy, sad, surprise, neutral, attention, recommended_minutes, session_date, session_duration)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)                      
                """, tuple(df_grouped.iloc[0]) + (final_recommendation, current_datetime, session_duration))  # 데이터프레임의 첫 번째 행을 튜플로 변환하고 추가 데이터와 결합
                conn.commit()                                                     # 데이터베이스 변경사항 저장

                # CSV 저장 (학습 데이터 축적)
                synthetic_path = "synthetic_sessions.csv"                         # 합성 데이터 파일 경로
                df_grouped["recommended_minutes"] = final_recommendation          # 데이터프레임에 최종 추천 시간 컬럼 추가
                if os.path.exists(synthetic_path):                               # CSV 파일이 이미 존재한다면
                    existing = pd.read_csv(synthetic_path)                       # 기존 데이터 읽기
                    updated = pd.concat([existing, df_grouped], ignore_index=True)  # 기존 데이터와 새 데이터 결합
                    updated.to_csv(synthetic_path, index=False)                  # 업데이트된 데이터를 CSV로 저장 (인덱스 제외)
                else:                                                            # CSV 파일이 없다면
                    df_grouped.to_csv(synthetic_path, index=False)               # 새 데이터를 CSV로 저장

                # 휴식 시간 시작
                st.session_state.is_break_time = True                            # 휴식 시간 상태 활성화
                st.session_state.break_start_time = time.time()                  # 휴식 시작 시간 기록
            
            # 상태 초기화
            st.session_state.total_elapsed_time = 0                              # 총 경과 시간 초기화
            st.session_state.pause_start_time = 0                                # 일시정지 시작 시간 초기화
            st.rerun()                                                           # 페이지 새로고침으로 상태 업데이트
            break                                                                # while 루프 종료
        
        else:                                                                    # 세션 시간이 아직 남아있다면
            # 측정 진행 중
            ret, frame = cap.read()                                              # 카메라에서 프레임 읽기 (ret: 성공 여부, frame: 이미지 데이터)
            if ret:                                                              # 프레임을 성공적으로 읽었다면
                # 일시정지가 아닐 때만 데이터 수집
                if not st.session_state.is_paused:                               # 일시정지 상태가 아니라면
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)             # OpenCV 이미지(BGR)를 RGB로 색상 공간 변환
                    h, w, _ = frame.shape                                        # 프레임의 높이, 너비, 채널 수 가져오기
                    results = face_mesh.process(img_rgb)                         # MediaPipe로 얼굴 랜드마크 검출

                    # 감정 인식
                    emotions = emotion_detector.detect_emotions(frame)           # FER로 프레임에서 감정 검출
                    if emotions:                                                 # 감정이 검출되었다면
                        top = emotions[0]                                        # 첫 번째 (가장 확실한) 감정 결과 선택
                        (x, y, w_box, h_box) = top["box"]                        # 얼굴 영역의 좌표와 크기 추출
                        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)  # 얼굴 주위에 녹색 사각형 그리기
                        emotion_text = ", ".join([f"{k}: {v:.2f}" for k, v in top["emotions"].items()])  # 감정별 점수를 문자열로 포맷팅
                        emotion_dict = top["emotions"]                           # 감정 딕셔너리 저장
                    else:                                                        # 감정이 검출되지 않았다면
                        emotion_text = "얼굴 감지 안됨"                            # 감지 실패 메시지
                        emotion_dict = {k: 0 for k in ['angry','disgust','fear','happy','sad','surprise','neutral']}  # 모든 감정을 0으로 설정

                    # 집중도 계산
                    attention = 0                                                # 집중도 초기값 0
                    if results.multi_face_landmarks:                             # 얼굴 랜드마크가 검출되었다면
                        landmarks = results.multi_face_landmarks[0]              # 첫 번째 얼굴의 랜드마크 선택
                        points = [(lm.x * w, lm.y * h) for lm in landmarks.landmark]  # 정규화된 좌표를 픽셀 좌표로 변환
                        left_eye = np.mean([points[33], points[133]], axis=0)    # 왼쪽 눈의 중심점 계산 (랜드마크 33, 133의 평균)
                        right_eye = np.mean([points[362], points[263]], axis=0)  # 오른쪽 눈의 중심점 계산 (랜드마크 362, 263의 평균)
                        eye_center = (left_eye + right_eye) / 2                  # 양 눈의 중심점 계산
                        screen_center = np.array([w / 2, h / 2])                 # 화면 중심점 계산
                        dist = np.linalg.norm(eye_center - screen_center)        # 눈 중심과 화면 중심 간 거리 계산
                        attention = max(0, 1 - dist / (w / 2))                   # 거리 기반 집중도 계산 (0~1 범위)

                    # 데이터 저장
                    timestamps.append(effective_elapsed)                         # 현재 경과 시간을 타임스탬프 리스트에 추가
                    attn_scores.append(attention)                                # 현재 집중도를 집중도 리스트에 추가
                    data.append({                                                # 현재 측정 데이터를 데이터 리스트에 추가
                        'timestamp': effective_elapsed,                          # 타임스탬프
                        **emotion_dict,                                          # 감정 딕셔너리 언패킹 (각 감정별 점수)
                        'attention': attention                                   # 집중도
                    })

                    # 세션 상태 업데이트
                    st.session_state.data = data                                 # 업데이트된 데이터를 세션에 저장
                    st.session_state.timestamps = timestamps                     # 업데이트된 타임스탬프를 세션에 저장
                    st.session_state.attn_scores = attn_scores                   # 업데이트된 집중도 점수를 세션에 저장

                    emotion_placeholder.markdown(f"**감정 상태**: {emotion_text}  \n**집중도**: `{attention:.2f}`")  # 감정과 집중도 정보를 화면에 표시
                else:                                                            # 일시정지 상태라면
                    emotion_placeholder.markdown("⏸️ **일시정지 중**")           # 일시정지 메시지 표시

                # 타이머 표시
                remaining_time = session_time * 60 - effective_elapsed           # 남은 시간 계산 (초 단위)
                minutes = int(remaining_time // 60)                              # 남은 분 계산
                seconds = int(remaining_time % 60)                               # 남은 초 계산
                status = "⏸️ 일시정지" if st.session_state.is_paused else "▶️ 측정 중"  # 현재 상태에 따른 상태 텍스트 설정
                timer_placeholder.markdown(f"**{status}**  \n남은 시간: `{minutes:02d}:{seconds:02d}`")  # 상태와 남은 시간을 화면에 표시

                # 영상 표시
                frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)           # OpenCV 프레임을 RGB로 변환 (Streamlit 표시용)
                frame_display = cv2.flip(frame_display, 1)                       # 프레임을 좌우 반전 (거울 효과)
                frame_placeholder.image(frame_display, channels="RGB")           # 변환된 프레임을 화면에 표시

                # 집중도 그래프 업데이트
                if timestamps and attn_scores:                                   # 타임스탬프와 집중도 점수가 있다면
                    fig, ax = plt.subplots()                                     # matplotlib 그래프 객체 생성
                    fig.set_size_inches(6, 4)                                    # 그래프 크기 설정 (6x4 인치)
                    ax.plot(timestamps, attn_scores)                             # 시간에 따른 집중도 변화를 선 그래프로 그리기
                    ax.set_ylim(0, 1)                                            # y축 범위를 0~1로 설정
                    ax.set_title("Real-time Attention")                          # 그래프 제목 설정
                    ax.set_xlabel("Time (seconds)")                              # x축 라벨 설정
                    ax.set_ylabel("Attention Score")                             # y축 라벨 설정
                    graph_placeholder.pyplot(fig, use_container_width=True)      # 그래프를 화면에 표시 (컨테이너 너비에 맞춤)

            else:                                                                # 프레임 읽기에 실패했다면
                st.error("카메라 오류")                                          # 에러 메시지 표시
                st.session_state.is_measuring = False                            # 측정 상태 해제
                if st.session_state.cap:                                         # 카메라 객체가 있다면
                    st.session_state.cap.release()                               # 카메라 리소스 해제
                    st.session_state.cap = None                                  # 카메라 객체를 None으로 설정
                break                                                            # while 루프 종료

# === 8. 카메라 정리 ===
if not st.session_state.is_measuring and 'cap' in st.session_state and st.session_state.cap is not None:  # 측정 중이 아니고 카메라 객체가 존재한다면
    st.session_state.cap.release()                                               # 카메라 리소스 해제
    st.session_state.cap = None                                                  # 카메라 객체를 None으로 설정하여 메모리 정리
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
