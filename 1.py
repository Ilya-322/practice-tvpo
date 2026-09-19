import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Загружаем встроенный датасет diabetes
diabetes = datasets.load_diabetes()

X = diabetes.data[:, [2]]
y = diabetes.target

X_train = X[:-20]
X_test = X[-20:]

y_train = y[:-20]
y_test = y[-20:]

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Коэффициент:", model.coef_[0])
print("MSE:", mean_squared_error(y_test, predictions))
print("R²:", r2_score(y_test, predictions))

plt.scatter(X_test, y_test, label="Реальные значения")
plt.plot(X_test, predictions, color="red", label="Линия регрессии")
plt.xlabel("Признак")
plt.ylabel("Целевая переменная")
plt.title("Ordinary Least Squares (Diabetes)")
plt.legend()
plt.show()