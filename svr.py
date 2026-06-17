import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


# =========================
# 데이터 불러오기
# =========================
df1 = pd.read_excel("2023.xlsx")
df2 = pd.read_excel("2024.xlsx")
df3 = pd.read_excel("2025.xlsx")

df = pd.concat([df1, df2, df3], ignore_index=True)

df = df.rename(columns={
    '누적강수량(mm)': 'rainfall',
    '일시': 'datetime'
})

df['datetime'] = pd.to_datetime(df['datetime'])

df = df.sort_values('datetime')
df = df.set_index('datetime')

# 30분 단위 리샘플링
df = df.resample('30min').mean()

df = df.reset_index()


# =========================
# 인공 폭우 데이터 생성
# =========================
np.random.seed(42)

num_events = 8

rain_indices = np.random.choice(
    len(df),
    size=num_events,
    replace=False
)

for idx in rain_indices:

    # 지속 시간
    duration = np.random.choice([6, 12, 24, 36])

    # 강한 강수량
    heavy_rain = np.random.choice([30, 50, 80, 100])

    for i in range(idx, min(idx + duration, len(df))):
        df.loc[i, 'rainfall'] = heavy_rain


# =========================
# 사용할 변수 선택
# =========================
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


# =========================
# 결측치 처리 및 정규화
# =========================
df = df.fillna(df.mean(numeric_only=True))

data = df[features].values

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)


# =========================
# 시계열 입력 생성
# =========================
X = []

n_past = 5

for i in range(n_past, len(scaled_data)):
    X.append(
        scaled_data[i - n_past:i].flatten()
    )

X = np.array(X)

print("X shape:", X.shape)


# =========================
# AutoEncoder 모델
# =========================
input_dim = X.shape[1]

input_layer = Input(shape=(input_dim,))

encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(8, activation='relu')(encoded)

decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(
    inputs=input_layer,
    outputs=decoded
)

autoencoder.compile(
    optimizer='adam',
    loss='mse'
)


# =========================
# 학습
# =========================
history = autoencoder.fit(
    X,
    X,
    epochs=30,
    batch_size=32,
    verbose=1
)


# =========================
# 복원 결과
# =========================
reconstructed = autoencoder.predict(X)

errors = np.mean(
    np.power(X - reconstructed, 2),
    axis=1
)

actual = X[:, 0]
predicted = reconstructed[:, 0]

mse = mean_squared_error(actual, predicted)

print("AutoEncoder MSE:", round(mse, 6))


# =========================
# 실제값 vs 복원값
# =========================
plt.figure(figsize=(10, 5))

plt.plot(actual[-300:], label='Actual')
plt.plot(predicted[-300:], label='Reconstructed')

plt.title("AutoEncoder Reconstruction (Long Heavy Rain)")
plt.xlabel("Time")
plt.ylabel("Scaled")

plt.legend()
plt.grid()

plt.show()


# =========================
# 산점도
# =========================
plt.figure(figsize=(6, 6))

plt.scatter(
    actual,
    predicted,
    alpha=0.3
)

min_val = min(actual.min(), predicted.min())
max_val = max(actual.max(), predicted.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color='red'
)

plt.title("AutoEncoder Performance")
plt.xlabel("Actual")
plt.ylabel("Reconstructed")

plt.grid()

plt.show()


# =========================
# 이상치 탐지
# =========================
threshold = np.percentile(errors, 95)

plt.figure(figsize=(10, 5))

plt.plot(
    errors,
    label='Reconstruction Error'
)

plt.axhline(
    threshold,
    color='red',
    linestyle='--',
    label='Threshold'
)

anomalies = np.where(errors > threshold)[0]

plt.scatter(
    anomalies,
    errors[anomalies],
    color='red',
    label='Anomaly'
)

plt.title("Anomaly Detection (Long Heavy Rain)")
plt.xlabel("Time")
plt.ylabel("Error")

plt.legend()
plt.grid()

plt.show()