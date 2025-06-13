# session_core1로부터 랜덤 포레스트 회귀 모델로 수정, 집중 상태 분석 및 추천 시간 계산 추가, 집중 상태별 시간 추천 로직 추가
# --- 필요한 라이브러리 불러오기 ---
import cv2  # OpenCV: 영상 처리 라이브러리
import time  # 시간 측정용 라이브러리
import numpy as np  # 수치 연산을 위한 라이브러리
import pandas as pd  # 데이터프레임으로 데이터 관리
from fer import FER  # FER: 감정 인식 라이브러리
import mediapipe as mp  # MediaPipe: 얼굴 메쉬 및 포즈 추정 등
from sklearn.ensemble import RandomForestRegressor  # 랜덤 포레스트 회귀 모델 (성능 향상)
import os  # 파일 존재 여부 확인용
import matplotlib.pyplot as plt  # 시각화 용도

# --- 감정 인식 모델 초기화 (MTCNN 사용 안 함으로 설정) ---
emotion_detector = FER(mtcnn=False) # 다중 작업 계단식 합성곱 신경(얼굴을 더 정확하게 감지하기 위해 사용하는 딥러닝 기반 얼굴 인식 알고리즘, 속도 느려짐)

# --- MediaPipe의 얼굴 메쉬 관련 초기화 ---
mp_face_mesh = mp.solutions.face_mesh  # 얼굴의 눈, 코, 입, 윤곽선 등 468개의 정밀한 점(랜드마크)을 추적하는 기능,얼굴의 자세한 구조를 실시간으로 파악
mp_drawing = mp.solutions.drawing_utils  # 얼굴에 찍힌 점(랜드마크)을 화면에 시각적으로 그려주는 도구
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) # MediaPipe 얼굴 메쉬 분석기를 실제로 생성(실시간 스트리밍 영상용 모드,얼굴 하나만 추적)

# --- 집중도 추정 함수 정의 ---
# 집중도(화면을 얼마나 바라보고 있는지)를 계산하는 함수
def estimate_attention(frame, landmarks, w, h): # 현재 영상 프레임, 얼굴의 468개 좌표가 담긴 데이터, 너비, 높이
    # 얼굴 랜드마크들을 (픽셀 좌표로) 변환
    points = [(lm.x * w, lm.y * h) for lm in landmarks.landmark]
    # MediaPipe가 제공하는 얼굴 랜드마크는 0~1 사이의 정규화된 좌표, 이것을 픽셀 단위로 변환하는 코드
    # 예를 들어, lm.x=0.5이고 w=640이라면 실제 좌표는 x=320
    #모든 랜드마크를 픽셀 위치로 바꿔서 points에 저장
    
    # 왼쪽 눈 중심 좌표 계산 (두 지점 평균)
    left_eye_center = np.mean([points[33], points[133]], axis=0) 
    # points[33]과 points[133]는 왼쪽 눈의 양 끝을 나타내는 랜드마크, 두 좌표의 x, y 평균을 계산
    
    # 오른쪽 눈 중심 좌표 계산
    right_eye_center = np.mean([points[362], points[263]], axis=0)
    
    # 두 눈의 중간 지점을 눈 중심으로 설정
    eye_center = (left_eye_center + right_eye_center) / 2 # 왼쪽 눈 중심과 오른쪽 눈 중심의 중간 지점, 두 눈 사이의 중앙, 즉 사용자의 시선 중심이라고 간주
    
    # 화면 중심 좌표 계산
    screen_center = np.array([w / 2, h / 2]) # 영상 화면의 정중앙 좌표를 계산
    
    # 눈 중심과 화면 중심 사이의 거리 계산
    dist = np.linalg.norm(eye_center - screen_center) # 벡터나 행렬의 크기(길이)를 계산할 때 사용, 주로 두 점 사이의 거리
    
    # 최대 거리 설정 (화면 너비의 절반)
    max_dist = w / 2
    
    # 거리에 비례하여 집중도 점수(0~1 사이) 계산 (멀어질수록 집중도 낮음)
    attention_score = max(0, 1 - dist / max_dist) # dist / max_dist → 화면에서 얼마나 벗어났는지를 0~1로 환산,
    # 1 - dist / max_dist → 벗어난 정도가 작을수록 높은 집중, max(0,x): 음수가 나오지 않게 처리
    return attention_score

# --- 감정 및 집중도 측정 + 실시간 시각화 함수 ---
def analyze_session(duration_minutes=1, output_file='session_data.csv'): # 측정할 시간(분) 기본값 1분,결과를 저장할 CSV 파일 이름
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # 웹캠 열기
    start_time = time.time()  # 측정 시작 시간 저장
    data = []  # 측정한 모든 데이터를 저장할 리스트
    timestamps = []  # 경과 시간 저장용 리스트
    attention_scores = []  # 집중도 점수만 저장할 리스트

    print(f"세션 시작: {duration_minutes}분 동안 측정 중...") # 시작 안내 출력: 몇 분 동안 측정을 시작하는지 사용자에게 알려줌

    # 지정된 시간 동안 반복
    while (time.time() - start_time) < duration_minutes * 60: # 현재 시간에서 시작 시간 빼서 경과 시간 계산 : 경과 시간이 duration_minutes(분) * 60(초)보다 작으면 계속 실행
        ret, frame = cap.read()  # ret은 읽기 성공 여부(True/False), frame은 읽은 영상 이미지
        if not ret:              # 프레임 읽기 실패하면 오류 출력 후 반복 종료
            print("웹캠 오류")
            break  # 실패 시 종료

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV는 기본적으로 BGR 형식,얼굴 인식 라이브러리에서는 RGB를 사용하므로 변환
        results = face_mesh.process(img_rgb)  # 얼굴 메쉬 추출

        # 얼굴을 인식한 다음, 감정 인식
        emotions = emotion_detector.detect_emotions(frame) # 현재 프레임에서 감정 분석 수행
        if emotions: # 얼굴이 있으면
            top_emotions = emotions[0]["emotions"]  # 화면에 감지된 첫번째 사람의 감정 분석 점수를 담은 딕셔너리(각 감정이 나타날 확률(0~1 사이))
            (x, y, w_box, h_box) = emotions[0]["box"]  # 얼굴 위치 좌표를 받아 사각형을 영상에 그림(초록색) 
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)  
        else:
            # 얼굴이 감지되지 않으면 모든 감정 점수 0 값 설정
            top_emotions = {k: 0 for k in ['angry','disgust','fear','happy','sad','surprise','neutral']}

        # 얼굴을 인식한 다음, 집중도(attention) 점수를 계산
        attention = 0  # 기본 집중도 초기화
        h, w, _ = frame.shape  # h = 영상 세로 픽셀 수, w = 가로 픽셀 수
        if results.multi_face_landmarks: # 얼굴 랜드마크가 있으면
            landmarks = results.multi_face_landmarks[0]  # 첫 번째 얼굴의 랜드마크 선택
            # mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_CONTOURS)  # 필요 시 랜드마크 그리기
            attention = estimate_attention(frame, landmarks, w, h)  # 집중도 점수 계산

        elapsed = time.time() - start_time  # 경과 시간 계산(웹캠 열렸을 때부터터)
        timestamps.append(elapsed)  # 시간 기록
        attention_scores.append(attention)  # 집중도 점수도 기록

        # 시간,감정,집중도 정보를 하나의 딕셔너리로 만들어서 data에 저장
        data.append({
            'timestamp': elapsed,
            **top_emotions,    # 딕셔너리를 ** 연산자로 펼쳐서 넣기
            'attention': attention
        }) 

        # 현재 영상 프레임에 집중도와 가장 높은 감정 출력
        info = f"Attn: {attention:.2f} | Emotions: {max(top_emotions, key=top_emotions.get)}" 
        # 집중도 소수점 둘째 자리까지 표시,감정 딕셔너리 중 가장 큰 값 이름 출력
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # 화면 좌측 상단에 흰색 글씨로 표시
        cv2.imshow("감정 + 집중도 측정", frame)  # 프레임 출력

        # 'q' 키 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # 웹캠 해제
    cv2.destroyAllWindows()  # 모든 opencv 창 닫기

    # 수집한 데이터로 데이터프레임 생성 후 CSV 파일로 저장
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    actual_duration = (time.time() - start_time) / 60  # 실제 측정 시간(분) 계산 및 출력
    print(f"측정 종료, 저장 완료: {output_file}")
    print(f"실제 시간: {actual_duration:.2f}분")
    return actual_duration # 실제 측정 시간 반환

# --- 쉬는 시간 타이머 ---
def break_timer(minutes=5):
    print(f"\n쉬는 시간 {minutes}분 시작...")
    time.sleep(minutes * 60)  # 주어진 시간 동안 대기(분 단위)
    print("쉬는 시간 종료!\n")

# --- 랜덤 포레스트 회귀 모델을 학습하여 다음 세션 추천 시간 예측 ---
def train_regression_model(data_path='session_data.csv', all_sessions_path='sessions_all.csv'): 
    # data_path와 all_sessions_path라는 파일 경로를 입력받아, 각각 최신 세션 데이터와 전체 세션 데이터를 저장하는 CSV 파일 경로
    df = pd.read_csv(data_path)  # 'session_data.csv' 파일을 읽어와 df라는 데이터프레임에 저장(가장 최근 세션)
    feature_cols = ['angry','disgust','fear','happy','sad','surprise','neutral','attention']
    
    # 각 감정 및 집중도 평균값 계산
    df_grouped = df[feature_cols].mean().to_frame().T # df에서 감정과 집중도 컬럼들만 뽑아서 각 컬럼별 평균값을 계산
    # .mean()은 각 열(column)의 평균을 계산, .to_frame()은 결과를 데이터프레임 형식으로 변환
    # .T는 행과 열을 뒤집어서(전치) 평균값들이 한 행(row)으로 만들어진 데이터프레임이 되도록 함
    
    # 집중 상태 분석 및 추천 시간 계산
    angry_score = df_grouped['angry'].iloc[0]
    neutral_score = df_grouped['neutral'].iloc[0]
    attention_score = df_grouped['attention'].iloc[0]
    
    # 집중 상태별 시간 추천 로직
    if angry_score > 0.12 and attention_score >= 0.5:  # 매우 집중 상태
        recommended_time = 25 + angry_score * 15 + attention_score * 10
        print(f"매우 집중 (Neutral: {neutral_score:.2f}, Angry: {angry_score:.2f}, Attention: {attention_score:.2f}) → 집중력 유지하며 시간 증가")
    elif neutral_score >= 0.6 and attention_score >= 0.5: # 보통 집중 상태
        recommended_time = max(20, 25 - neutral_score * 8)
        print(f"보통 집중 (Neutral: {neutral_score:.2f}, Angry: {angry_score:.2f}, Attention: {attention_score:.2f}) → 더 높은 집중 유도를 위해 시간 단축")
    else:  # 산만한 상태
        recommended_time = 30.0
        print(f"산만함 (Neutral: {neutral_score:.2f}, Angry: {angry_score:.2f}, Attention: {attention_score:.2f}) → 차분히 앉아있기 위해 긴 시간 권장")
    
    # 추천 시간 범위 제한 (20-50분)
    recommended_time = np.clip(recommended_time, 20, 50)
    df_grouped['recommended_time'] = recommended_time

    # 과거 세션 데이터 있으면 불러와서 합치기
    if os.path.exists(all_sessions_path): # sessions_all.csv 파일이 컴퓨터에 존재하는지 확인
        prev = pd.read_csv(all_sessions_path) # 파일이 존재하면 이전 모든 세션 데이터를 읽어서 prev에 저장
        full = pd.concat([prev, df_grouped], ignore_index=True) # prev 데이터프레임과 이번 세션의 평균값(df_grouped)을 하나로 이어붙여서 full에 저장
        # ignore_index=True는 새로 합친 데이터프레임의 인덱스를 0부터 다시 매김
    else: # 만약 이전 데이터 파일이 없으면, 이번 세션 데이터만 full로 사용
        full = df_grouped

    full.to_csv(all_sessions_path, index=False)     # 지금까지 누적된 전체 세션 데이터를 sessions_all.csv 파일로 저장
                                                    # index=False는 행 번호(인덱스)를 파일에 저장하지 말라는 뜻

    if len(full) > 3:  # full 데이터프레임 행(row) 수가 3개 초과이면(즉, 데이터가 충분하면) 회귀 모델 학습을 시작
        X = full[feature_cols]  # full에서 감정과 집중도 컬럼들만 뽑아서 X에 저장, X가 모델의 입력 데이터(특성 행렬)
        y = full['recommended_time']  # recommended_time 컬럼을 y에 저장, 목표 값
        model = RandomForestRegressor(n_estimators=100, random_state=42)  # 랜덤 포레스트로 변경 (성능 향상)
        model.fit(X, y)  #  입력 특성들로부터 추천 시간을 예측하는 규칙을 찾는 과정
        new_time = model.predict(df_grouped[feature_cols])[0]  # 방금 측정한 세션 평균 감정/집중도(df_grouped[feature_cols])를 모델에 넣어 다음 추천 시간을 예측, 결과는 배열 형태로 나오는데, [0]을 써서 첫 번째 값만 꺼내서 new_time에 저장
        print(f"모델 예측 추천 시간: {round(new_time,2)}분") # 예측한 추천 시간을 소수점 둘째 자리까지 반올림하여 출력
        return max(20, round(new_time, 2))  # 추천 시간이 너무 짧으면 의미 없으니까, 최소 20분으로 보장해서 반환
    else:
        print(f"데이터 부족 → 규칙 기반 추천: {recommended_time:.2f}분") # 데이터가 3개 이하로 부족하면 모델을 학습하지 않고, 규칙 기반 추천 시간을 반환
        return round(recommended_time, 2)  

# --- 사용자에게 다음 세션 시간 입력받기 ---
def ask_next_duration(default_duration):
    try:
        # input() 함수로 사용자에게 텍스트를 입력받음
        user_input = input(f"\n다음 뽀모도로 시간을 입력하세요 (기본값 {default_duration}분, 최소 0.5분): ").strip()
        if user_input == "":
            return default_duration  # 입력 없으면 기본값
        val = float(user_input)  # 숫자로 변환
        if val < 0.5: # 만약 입력한 시간이 0.5분 미만이면, 경고 메시지를 출력하고 기본값을 반환
            print("❗ 최소 0.5분 이상 입력해야 합니다.")
            return default_duration 
        return val  # 입력값이 정상(0.5 이상 숫자)이면, 그 값을 반환
    except: # float() 변환 중 오류가 나면, 에러 메시지를 출력하고 기본값을 반환
        print("❗ 잘못된 입력입니다. 기본값으로 진행합니다.")
        return default_duration  # 오류 시 기본값 사용

# --- 전체 실행 메인 함수 (하나의 세션 진행) ---
def run_pomodoro_session(initial_duration=1, break_duration=0.17):
    # 현재 세션 실행 (기본 1분), 집중 시간 동안 감정 + 집중도 측정하고, 데이터를 session_data.csv에 저장
    actual_duration = analyze_session(duration_minutes=initial_duration, output_file='session_data.csv')
    
    # 방금 측정이 끝났으면, 설정한 시간만큼 쉬는 시간
    break_timer(minutes=break_duration)

    # 방금 저장된 데이터를 바탕으로 머신러닝 회귀 모델이 다음 집중 시간을 예측, 추천된 시간이 recommended_duration 변수에 저장
    recommended_duration = train_regression_model(data_path='session_data.csv')
    print(f"\n📈 추천된 다음 뽀모도로 시간: {recommended_duration}분") # 예측된 추천 시간을 사용자에게 출력

    # 추천된 시간 그대로 쓸지, 아니면 사용자가 직접 새로 입력할지를 물어봄
    next_duration = ask_next_duration(recommended_duration) # 사용자가 입력한 시간이 next_duration 변수에 저장
    print(f"⏱ 다음 세션 시간: {next_duration}분으로 설정됨.\n")
    
    return next_duration  # 다음 세션에서 사용할 시간(next_duration)을 반환해서 외부에서도 쓸 수 있게 함