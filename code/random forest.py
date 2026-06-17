# =========================================
# AutoEncoder 기반 강우량 예측
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# =========================================
# 데이터 불러오기
# =========================================

df1 = pd.read_excel("C:/Users/konyang/Desktop/AI강우량/남해 23~25년도/test2 2023.xlsx")
df2 = pd.read_excel("C:/Users/konyang/Desktop/AI강우량/남해 23~25년도/test2 2024.xlsx")
df3 = pd.read_excel("C:/Users/konyang/Desktop/AI강우량/남해 23~25년도/test2 2025.xlsx")

df = pd.concat([df1, df2, df3], ignore_index=True)

df['일시'] = pd.to_datetime(df['일시'])

# =========================================
# 일별 강우량 생성
# =========================================

df['날짜'] = df['일시'].dt.date

daily_df = df.groupby('날짜')['누적강수량(mm)'].max().reset_index()

data = daily_df[['누적강수량(mm)']]

# =========================================
# 정규화
# =========================================

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)

# =========================================
# 시계열 데이터 생성
# 과거 7일 → 다음날
# =========================================

n_past = 7

X = []
y = []

for i in range(n_past, len(scaled_data)):
    X.append(
        scaled_data[i-n_past:i].flatten()
    )

    y.append(
        scaled_data[i,0]
    )

X = np.array(X)
y = np.array(y)

print(X.shape)
print(y.shape)

# =========================================
# AutoEncoder 기반 예측 모델
# =========================================

input_dim = X.shape[1]

input_layer = Input(shape=(input_dim,))

encoded = Dense(
    32,
    activation='relu'
)(input_layer)

encoded = Dense(
    16,
    activation='relu'
)(encoded)

encoded = Dense(
    8,
    activation='relu'
)(encoded)

output = Dense(
    1,
    activation='linear'
)(encoded)

model = Model(
    inputs=input_layer,
    outputs=output
)

# =========================================
# 컴파일
# =========================================

model.compile(
    optimizer='adam',
    loss='mse'
)

# =========================================
# 학습
# =========================================

history = model.fit(
    X,
    y,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# =========================================
# 예측
# =========================================

pred = model.predict(X)

# =========================================
# 역정규화
# =========================================

real = scaler.inverse_transform(
    y.reshape(-1,1)
)

pred = scaler.inverse_transform(
    pred
)

# =========================================
# 성능 평가
# =========================================

mse = mean_squared_error(
    real,
    pred
)

rmse = np.sqrt(mse)

print("MSE :", mse)
print("RMSE :", rmse)

# =========================================
# 그래프
# =========================================

dates = daily_df['날짜'][n_past:].astype(str)

plt.figure(figsize=(18,6))

plt.plot(
    dates,
    real,
    label='Real Rainfall',
    color='tab:blue'
)

plt.plot(
    dates,
    pred,
    label='AutoEncoder Prediction',
    color='tab:orange'
)

step = 15

plt.xticks(
    np.arange(0, len(dates), step),
    dates[::step],
    rotation=45
)


plt.xticks(
    np.arange(0, len(dates), step),
    dates[::step],
    rotation=45
)

plt.xlabel("Date")
plt.ylabel("Rainfall (mm)")
plt.title("AutoEncoder Rainfall Prediction")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()