# =========================================
# Random Forest 강우량 예측
# =========================================

# 1. 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# =========================================
# 2. 데이터 불러오기
# =========================================

df1 = pd.read_excel("C:/Users/a0109/OneDrive/바탕 화면/AI강우량/통영 2023 최종.xlsx")
df2 = pd.read_excel("C:/Users/a0109/OneDrive/바탕 화면/AI강우량/통영 2024 최종.xlsx")
df3 = pd.read_excel("C:/Users/a0109/OneDrive/바탕 화면/AI강우량/통영 2025 최종.xlsx")

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
# =========================================

df['날짜'] = df['일시'].dt.date

daily_df = df.groupby(
    '날짜'
)['누적강수량(mm)'].mean()

daily_df = daily_df.reset_index()

# =========================================
# 6. 사용할 데이터
# =========================================

data = daily_df[['누적강수량(mm)']]

# =========================================
# 7. 정규화
# =========================================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)

# =========================================
# 8. 시계열 데이터 생성
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
# 9. Random Forest 모델 생성
# =========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# =========================================
# 10. 모델 학습
# =========================================

model.fit(X, y)

# =========================================
# 11. 예측
# =========================================

predictions = model.predict(X)

# =========================================
# 12. 역정규화
# =========================================

real = scaler.inverse_transform(
    y.reshape(-1,1)
)

pred = scaler.inverse_transform(
    predictions.reshape(-1,1)
)

# =========================================
# 13. 성능 평가
# =========================================

mse = mean_squared_error(
    real,
    pred
)

print("Random Forest MSE:", mse)

# =========================================
# 14. 그래프
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
    label='RF Prediction'
)

step = 15

plt.xticks(
    np.arange(0, len(dates), step),
    dates[::step],
    rotation=45
)
plt.xlabel("Date")
plt.ylabel("Rainfall (mm)")

plt.title("Random Forest Rainfall Prediction")

plt.legend()

plt.grid()

plt.show()