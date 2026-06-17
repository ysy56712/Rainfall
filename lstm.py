# =========================================
# LSTM 강우량 예측 (최종 수정)
# =========================================

# 1. 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =========================================
# 2. 데이터 불러오기
# =========================================

df1 = pd.read_excel("C:/Users/konyang/Desktop/AI강우량/남해 23~25년도/test2 2023.xlsx")
df2 = pd.read_excel("C:/Users/konyang/Desktop/AI강우량/남해 23~25년도/test2 2024.xlsx")
df3 = pd.read_excel("C:/Users/konyang/Desktop/AI강우량/남해 23~25년도/test2 2025.xlsx")


# 데이터 합치기
df = pd.concat(
    [df1, df2, df3],
    ignore_index=True
)

# =========================================
# 3. 날짜 형식 변환
# =========================================

df['일시'] = pd.to_datetime(df['일시'])

# =========================================
# 4. 결측치 처리
# =========================================

df = df.fillna(
    df.select_dtypes(include=np.number).mean()
)

# =========================================
# 5. 일별 강우량 생성
# 핵심: mean 말고 max 사용
# =========================================

df['날짜'] = df['일시'].dt.date

daily_df = df.groupby(
    '날짜'
)['누적강수량(mm)'].max().reset_index()

# =========================================
# 6. 사용할 데이터
# =========================================

data = daily_df[['누적강수량(mm)']]

# =========================================
# 7. 정규화
# =========================================

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)

# =========================================
# 8. 시계열 데이터 생성
# 최근 7일 -> 다음날 예측
# =========================================

X = []
y = []

n_past = 7

for i in range(n_past, len(scaled_data)):

    X.append(
        scaled_data[i-n_past:i, 0]
    )

    y.append(
        scaled_data[i, 0]
    )

X = np.array(X)
y = np.array(y)

# =========================================
# 9. LSTM 입력 형태 변환
# =========================================

X = X.reshape(
    X.shape[0],
    X.shape[1],
    1
)

# =========================================
# 10. 모델 생성
# =========================================

model = Sequential()

model.add(
    LSTM(
        64,
        input_shape=(X.shape[1], 1)
    )
)

model.add(Dense(1))

# =========================================
# 11. 컴파일
# =========================================

model.compile(
    optimizer='adam',
    loss='mse'
)

# =========================================
# 12. 학습
# =========================================

model.fit(
    X,
    y,
    epochs=100,
    batch_size=16,
    verbose=1
)

# =========================================
# 13. 예측
# =========================================

predictions = model.predict(X)

# =========================================
# 14. 역정규화
# =========================================

real = scaler.inverse_transform(
    y.reshape(-1,1)
)

pred = scaler.inverse_transform(
    predictions
)

# =========================================
# 15. 성능 평가
# =========================================

mse = mean_squared_error(
    real,
    pred
)

print("LSTM MSE:", mse)

# =========================================
# 16. 그래프
# =========================================

dates = daily_df['날짜'].astype(str)[n_past:]

plt.figure(figsize=(18,6))

plt.plot(
    dates,
    real,
    label='Real Rainfall'
)

plt.plot(
    dates,
    pred,
    label='LSTM Prediction'
)

step = 15

plt.xticks(
    np.arange(0, len(dates), step),
    dates[::step],
    rotation=45
)

plt.xlabel("Date")
plt.ylabel("Rainfall (mm)")
plt.title("LSTM Rainfall Prediction")

plt.legend()
plt.grid()

plt.show()