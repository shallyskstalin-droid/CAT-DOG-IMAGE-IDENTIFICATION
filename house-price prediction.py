import pandas as pd
df = pd.read_csv("Housing.csv", encoding="ISO-8859-1")
df.head()
df.info()
df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded.head()
X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape)
print(X_test.shape)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred[:5]
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
