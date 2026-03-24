import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'D:\NITJ ACADEMICS\OPENLEARN\DailyDelhiClimateTest.csv')
print(df.head())
print("Missing values in the dataset.")
print(df.isnull().sum())

class MeraMLR:
    def __init__(self):
        self.beta = None  
    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train).reshape(-1, 1)        
        ones = np.ones((X_train.shape[0], 1))
        X_b = np.hstack((ones, X_train))      
        self.beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_train
    def predict(self, X_test):
        X_test = np.array(X_test)
        ones = np.ones((X_test.shape[0], 1))
        X_b = np.hstack((ones, X_test))
        return X_b @ self.beta
    def r2_score(self, y_true, y_pred):
        y_true = np.array(y_true).reshape(-1, 1)
        y_pred = np.array(y_pred)
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["meantemp"] = pd.to_numeric(df["meantemp"], errors="coerce")
df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")
df["meanpressure"] = pd.to_numeric(df["meanpressure"], errors="coerce")
df = df.dropna()
df["day_index"] = (df["date"] - df["date"].min()).dt.days
X = df[["day_index", "humidity", "wind_speed", "meanpressure"]]
y = df["meantemp"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MeraMLR()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = model.r2_score(y_test, y_pred)
print("R2 Score:", r2)

X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.2, random_state=42)
model1 = LinearRegression()
model1.fit(X1_train, y1_train)
y1_pred1 = model1.predict(X1_test)
r2_split = r2_score(y1_test, y1_pred1)
print("R2 Score( Inbuilt ):", r2_split)
