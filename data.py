
import pandas as pd
import numpy as np
import os



# --- 가상 데이터 생성 ---
path="synthetic_sessions.csv"
if not os.path.exists(path):
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "angry": np.random.beta(2, 5, n),
        "disgust": np.random.beta(1, 8, n),
        "fear": np.random.beta(2, 6, n),
        "happy": np.random.beta(5, 2, n),
        "sad": np.random.beta(2, 5, n),
        "surprise": np.random.beta(2, 4, n),
        "neutral": np.random.beta(4, 3, n),
        "attention": np.random.uniform(0.4, 1.0, n)
    })
    weights = np.array([0.5, 0.2, 0.3, -0.7, 0.6, 0.4, -0.2, 1.2])
    noise = np.random.normal(0, 2, n)
    df["recommended_minutes"] = 30 + df[df.columns].values @ weights + noise
    df["recommended_minutes"] = np.clip(df["recommended_minutes"], 20, 60)
    df.to_csv(path, index=False)
    
# 기본 data 파일로부터 angry,sad,fear를 집중 표정으로 간주한 수정된 data 파일
# pandas와 numpy는 데이터 다루는 데 필요한 파이썬 라이브러리
import pandas as pd
import numpy as np
import os  # 파일 경로를 확인하거나 다룰 때 사용

# --- 가상 데이터 생성 ---
# CSV 파일 이름 설정
path = "synthetic_sessions.csv"

# 만약 현재 폴더에 이 CSV 파일이 없다면, 아래 코드를 실행합니다.
if not os.path.exists(path):

    # 코드를 실행할 때마다 동일한 무작위 숫자가 생성되도록 시드 고정 (재현성 확보)
    np.random.seed(42)

    # 가상의 데이터 개수 설정 (예: 200개의 데이터 샘플 생성)
    n = 200

    # 감정(emotion)과 집중(attention) 정도를 나타내는 데이터프레임 생성
    df = pd.DataFrame({
        # 각각 감정의 강도를 0~1 사이의 숫자로 생성 (베타분포를 따름)
        # 베타분포의 모양에 따라 어떤 감정은 낮은 값 쪽으로, 어떤 감정은 높은 값 쪽으로 치우치게 됨
        # 0~1 사이의 숫자 200개를 베타분포에 따라 뽑음  (1,1) : 균등분포, (a>1,b>1) : 중앙 근처 값들이 많이 나옴, 
        # (a<1,b<1) : 0과 1 근처 값들이 많이 나옴, (a < b) : 0 쪽에 더 몰림 , (a > b) : 1 쪽에 더 몰림
        "angry": np.random.beta(3, 4, n),            # 매우 집중할 때 높아짐 (중간~높은 값 분포)
        "disgust": np.random.beta(1, 9, n),          # 기본적으로 낮게 유지 (거의 0에 가까움)
        "fear": np.random.beta(3, 4, n),             # 매우 집중할 때 높아짐 (중간~높은 값 분포)
        "happy": np.random.beta(1, 9, n),            # 기본적으로 낮게 유지 (거의 0에 가까움)
        "sad": np.random.beta(3, 4, n),              # 매우 집중할 때 높아짐 (중간~높은 값 분포)
        "surprise": np.random.beta(1, 9, n),         # 기본적으로 낮게 유지 (거의 0에 가까움)
        "neutral": np.random.beta(4, 3, n),          # 보통 집중할 때 높아짐 (높은 값 분포)
        "attention": np.random.uniform(0.3, 0.95, n) # 집중도 범위 (산만:0.3~0.5, 보통:0.5~0.7, 높음:0.7~0.8)
    })

    # 집중 상태에 따른 추천 시간 가중치 설정
    # [angry, disgust, fear, happy, sad, surprise, neutral, attention]
    weights = np.array([
        1.6,    # angry: 증가
        0.0,    # disgust: 의미 없음
        1.6,    # fear: 증가
        0.0,    # happy: 감소
        1.6,    # sad: 증가
        0.0,    # surprise: 의미 없음
        -0.5,   # neutral: 보통 집중 → 시간 감소 (더 높은 집중 유도)
        0.8     # attention: 시선 집중도
    ])
    
    # 추천 시간에 현실적인 오차(잡음)를 넣기 위한 정규분포 노이즈(평균 0, 표준편차 1.5)
    noise = np.random.normal(0, 1.5, n) # 노이즈를 조금 줄여서 더 안정적인 추천

    # 각 샘플마다 (감정 + 집중도) * 가중치 + 노이즈를 더해 '추천 시간' 계산
    df["recommended_minutes"] = 25 + df[df.columns].values @ weights + noise # 기본값 25분
    
    # 추천 시간 범위 조정: 최소 20분, 최대 50분
    # 산만한 상태(attention 낮음) → 30분 
    # 매우 집중(angry 높음, attention 높음) → 점진적 증가
    # 보통 집중(neutral 높음) → 조금 감소하여 더 높은 집중 유도
    df["recommended_minutes"] = np.clip(df["recommended_minutes"], 20, 50)
    
    # 특별한 경우 처리: attention이 매우 낮으면 무조건 30분
    low_attention_mask = df["attention"] < 0.55
    df.loc[low_attention_mask, "recommended_minutes"] = np.clip(
        df.loc[low_attention_mask, "recommended_minutes"], 30, 50
    )

    # 최종 데이터를 CSV 파일로 저장 (인덱스는 저장하지 않음)
    df.to_csv(path, index=False)
