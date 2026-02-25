import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class MeraLR:
    def __init__(self):
        self.m = None
        self.b = None
        
    def fit(self, X_train, y_train):
        X_train = np.array(X_train).flatten()
        y_train = np.array(y_train)        
        x_mean = np.mean(X_train)
        y_mean = np.mean(y_train)        
        num = np.sum((X_train - x_mean) * (y_train - y_mean))
        den = np.sum((X_train - x_mean) ** 2)        
        self.m = num / den
        self.b = y_mean - self.m * x_mean
        
    def predict(self, X_test):
        X_test = np.array(X_test).flatten()
        return self.m * X_test + self.b

    def r2_score(self, y_true, y_pred):
        y_true = np.array(y_true)
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

df = pd.read_csv("daily-minimum-temperatures-in-me.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Daily minimum temperatures"] = pd.to_numeric(df["Daily minimum temperatures"], errors="coerce") 
df = df.dropna()
df["day_index"] = (df["Date"] - df["Date"].min()).dt.days

X = df[["day_index"]]
y = df["Daily minimum temperatures"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = MeraLR()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = model.r2_score(y_test, y_pred)

print(f"Slope (m): {model.m}")
print(f"Intercept (b): {model.b}")
print(f"R2 Score: {r2}")
