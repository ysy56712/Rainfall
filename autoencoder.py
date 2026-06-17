import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


# ============================
# 데이터 불러오기
# ============================
df1 = pd.read_excel("to 2023.xlsx")
df2 = pd.read_excel("to 2024.xlsx")
df3 = pd.read_excel("to 2025.xlsx")

df = pd.concat([df1, df2, df3], ignore_index=True)

df = df.rename(columns={
    '누적강수량(mm)': 'rainfall',
    '일시': 'datetime'
})

df['datetime'] = pd.to_datetime(df['datetime'])

df = df.sort_values('datetime')
df = df.set_index('datetime')

df = df.resample('30min').mean()
df = df.reset_index()


# ============================
# 인공 폭우 생성
# ============================
np.random.seed(42)

num_events = 8

rain_indices = np.random.choice(
    len(df),
    size=num_events,
    replace=False
)

for idx in rain_indices:

    duration = np.random.choice([6, 12, 24, 36])
    heavy_rain = np.random.choice([30, 50, 80, 100])

    for i in range(idx, min(idx + duration, len(df))):
        df.loc[i, 'rainfall'] = heavy_rain


# ============================
# 변수 선택
# ============================
features = ['rainfall']

for col in [
    '평균기온(°C)',
    '습도(%)',
    '풍속(m/s)',
    '해면기압(hPa)',
    '현지기압(hPa)'
]:
    if col in df.columns:
        features.append(col)

print("사용 변수:", features)


# ============================
# 결측치 처리
# ============================
df = df.fillna(df.mean(numeric_only=True))


# ============================
# 스케일링
# ============================
data = df[features].values

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)


# ============================
# 입력 데이터 생성
# ============================
X = []

n_past = 20

for i in range(n_past, len(scaled_data)):

    window = scaled_data[i - n_past:i].copy()

    diff = np.diff(
        window[:, 0],
        prepend=window[0, 0]
    ).reshape(-1, 1)

    window = np.hstack([window, diff])

    X.append(window.flatten())

X = np.array(X)

print("X shape:", X.shape)


# ============================
# AutoEncoder 모델
# ============================
input_dim = X.shape[1]

input_layer = Input(shape=(input_dim,))

encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(
    inputs=input_layer,
    outputs=decoded
)

autoencoder.compile(
    optimizer='adam',
    loss='mse'
)


# ============================
# 학습
# ============================
autoencoder.fit(
    X,
    X,
    epochs=30,
    batch_size=32,
    verbose=1
)


# ============================
# 예측
# ============================
reconstructed = autoencoder.predict(X)

actual = X[:, -len(features) - 1]
predicted = reconstructed[:, -len(features) - 1]


# ============================
# 역정규화
# ============================
def inverse_rainfall(series, scaler):
    return series * scaler.scale_[0] + scaler.mean_[0]


actual_inv = inverse_rainfall(actual, scaler)
pred_inv = inverse_rainfall(predicted, scaler)


# ============================
# 성능 평가
# ============================
mse = mean_squared_error(
    actual_inv,
    pred_inv
)

print("MSE:", int(mse))


# ============================
# 실제값 vs 복원값
# ============================
plt.figure(figsize=(10, 5))

plt.plot(
    actual_inv[-300:],
    label='Actual'
)

plt.plot(
    pred_inv[-300:],
    label='Reconstructed'
)

plt.legend()
plt.grid()

plt.show()


# ============================
# 산점도
# ============================
plt.figure(figsize=(6, 6))

plt.scatter(
    actual_inv,
    pred_inv,
    alpha=0.3
)

m = min(actual_inv.min(), pred_inv.min())
M = max(actual_inv.max(), pred_inv.max())

plt.plot(
    [m, M],
    [m, M],
    color='red'
)

plt.grid()

plt.show()


# ============================
# 이상치 탐지
# ============================
errors = np.mean(
    (X - reconstructed) ** 2,
    axis=1
)

threshold = np.percentile(
    errors,
    99
)

plt.figure(figsize=(10, 5))

plt.plot(errors)

plt.axhline(
    threshold,
    color='red'
)

anomalies = np.where(
    errors > threshold
)[0]

plt.scatter(
    anomalies,
    errors[anomalies],
    color='red'
)

plt.grid()

plt.show()