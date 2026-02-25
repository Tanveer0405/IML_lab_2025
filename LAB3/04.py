import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class MeraMultiLR:
    def __init__(self):
        self.w0 = None
        self.w1 = None
        self.w2 = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))
        W = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.w0 = W[0]
        self.w1 = W[1]
        self.w2 = W[2]

    def predict(self, X):
        X = np.array(X)
        return self.w0 + self.w1 * X[:, 0] + self.w2 * X[:, 1]

    def r2_score(self, y_true, y_pred):
        y_true = np.array(y_true)
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)


df = pd.read_csv("daily-minimum-temperatures-in-me - Copy.csv")
df.columns = df.columns.str.strip()
df = df.drop(columns=["Unnamed: 2", "Unnamed: 3"], errors="ignore")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Daily minimum temperatures"] = pd.to_numeric(df["Daily minimum temperatures"], errors="coerce")
df["Humidity"] = pd.to_numeric(df["Humidity"], errors="coerce")

df = df.dropna()
df["day_index"] = (df["Date"] - df["Date"].min()).dt.days

X = df[["day_index", "Humidity"]]
y = df["Daily minimum temperatures"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = MeraMultiLR()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = model.r2_score(y_test, y_pred)

print("w0 (Intercept):", model.w0)
print("w1 (day_index):", model.w1)
print("w2 (Humidity):", model.w2)
print("R2 Score:", r2)
