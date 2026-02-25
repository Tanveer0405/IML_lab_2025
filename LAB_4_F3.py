import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model1 = LinearRegression()
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
r2_split = r2_score(y_test, y_pred1)
print("R2 Score (Train-Test Split):", r2_split)

model2 = LinearRegression()
model2.fit(X, y)
y_pred2 = model2.predict(X)
r2_full = r2_score(y, y_pred2)
print("R2 Score (Full Data):", r2_full)