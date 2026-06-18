import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D,
    GRU, Dense, Dropout,
    BatchNormalization, Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score
)

# =========================
# 설정
# =========================
n_past = 14
threshold = 50

# =========================
# 데이터 불러오기
# =========================
df1 = pd.read_csv("yeosu23.csv", encoding="utf-8-sig")
df2 = pd.read_csv("yeosu24.csv", encoding="utf-8-sig")
df3 = pd.read_csv("yeosu25.csv", encoding="utf-8-sig")

df = pd.concat([df1, df2, df3], ignore_index=True)

df.columns = [c.strip() for c in df.columns]

df["일시"] = pd.to_datetime(df["일시"], errors="coerce")
df["날짜"] = df["일시"].dt.date

df["누적강수량(mm)"] = pd.to_numeric(
    df["누적강수량(mm)"],
    errors="coerce"
)

daily_df = (
    df.groupby("날짜")["누적강수량(mm)"]
    .max()
    .reset_index()
    .dropna()
)

# =========================
# 스케일링
# =========================
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    daily_df[["누적강수량(mm)"]]
)

dates = daily_df["날짜"].astype(str).values

# =========================
# 시퀀스 생성
# =========================
X = []
y = []

for i in range(n_past, len(scaled_data)):
    X.append(
        scaled_data[i - n_past:i].flatten()
    )
    y.append(
        scaled_data[i, 0]
    )

X = np.array(X)
y = np.array(y)

dates = dates[n_past:]

# =========================
# LightGBM
# =========================
feature_names = [
    f"lag_{i+1}"
    for i in range(n_past)
]

X_df = pd.DataFrame(
    X,
    columns=feature_names
)

lgb_model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)

lgb_model.fit(X_df, y)

lgb_pred = scaler.inverse_transform(
    lgb_model.predict(X_df).reshape(-1, 1)
)

real = scaler.inverse_transform(
    y.reshape(-1, 1)
)

lgb_mse = mean_squared_error(
    real,
    lgb_pred
)

print("===== LightGBM =====")
print("MSE:", lgb_mse)

# =========================
# CNN + GRU
# =========================
X_seq = X.reshape(
    X.shape[0],
    n_past,
    1
)

tf.keras.backend.clear_session()

inp = Input(shape=(n_past, 1))

x = Conv1D(
    64,
    kernel_size=3,
    activation="relu",
    padding="same"
)(inp)

x = BatchNormalization()(x)

x = Conv1D(
    64,
    kernel_size=3,
    activation="relu",
    padding="same"
)(x)

x = BatchNormalization()(x)

x = MaxPooling1D(pool_size=2)(x)

x = Dropout(0.2)(x)

x = Bidirectional(
    GRU(
        64,
        return_sequences=True
    )
)(x)

x = Dropout(0.2)(x)

x = GRU(
    32,
    return_sequences=False
)(x)

x = Dropout(0.2)(x)

x = Dense(
    32,
    activation="relu"
)(x)

out = Dense(
    1,
    activation="sigmoid"
)(x)

model = Model(inp, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse"
)

model.fit(
    X_seq,
    y,
    epochs=1000,
    batch_size=8,
    validation_split=0.2,
    callbacks=[
        EarlyStopping(
            patience=50,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            patience=20,
            factor=0.5,
            min_lr=1e-6
        )
    ],
    verbose=1
)

cnn_pred = scaler.inverse_transform(
    model.predict(X_seq)
)

cnn_mse = mean_squared_error(
    real,
    cnn_pred
)

print("===== CNN + GRU =====")
print("MSE:", cnn_mse)

# =========================
# 결과 그래프
# =========================
plt.figure(figsize=(18, 6))

plt.plot(
    dates,
    real,
    label="Real",
    color="blue"
)

plt.plot(
    dates,
    cnn_pred,
    label="CNN+GRU",
    color="orange"
)

plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()