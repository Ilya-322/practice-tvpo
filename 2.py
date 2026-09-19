import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Загружаем данные
df = pd.read_csv("bottle.csv")

# Оставляем нужные столбцы и удаляем пропуски
df = df[["Salnty", "T_degC"]].dropna()

X = df[["Salnty"]]
y = df["T_degC"]

# Делим выборку
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Обучаем модель
model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

# Метрики
print("R²:", r2_score(y_test, pred))
print("MAE:", mean_absolute_error(y_test, pred))
print("MSE:", mean_squared_error(y_test, pred))
print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)

# График
plt.scatter(X_test, y_test, s=2)
plt.plot(X_test, pred, color="red")
plt.xlabel("Salinity")
plt.ylabel("Temperature")
plt.title("Linear Regression (CalCOFI)")
plt.show()