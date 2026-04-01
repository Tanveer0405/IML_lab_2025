# SVM (HARDCODE)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

class SVM_HC:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.where(linear_output >= 0, 1, 0)

df_d = pd.read_csv('diabetes_dataset.csv')
df_d.fillna(df_d.mean(numeric_only=True), inplace=True)
for col in df_d.columns:
    if df_d[col].dtype == 'object':
        df_d[col] = LabelEncoder().fit_transform(df_d[col])
X_d = df_d.iloc[:, :-1].values
y_d = df_d.iloc[:, -1].values
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.2, random_state=42)
scaler_d = StandardScaler()
X_train_d = scaler_d.fit_transform(X_train_d)
X_test_d = scaler_d.transform(X_test_d)
model_d = SVM_HC()
model_d.fit(X_train_d, y_train_d)
y_pred_d = model_d.predict(X_test_d)
print("Diabetes", accuracy_score(y_test_d, y_pred_d))

df_s = pd.read_csv('Social_Network_Ads.csv')
df_s.fillna(df_s.mean(numeric_only=True), inplace=True)
for col in df_s.columns:
    if df_s[col].dtype == 'object':
        df_s[col] = LabelEncoder().fit_transform(df_s[col])
X_s = df_s.iloc[:, :-1].values
y_s = df_s.iloc[:, -1].values
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_s, y_s, test_size=0.2, random_state=42)
scaler_s = StandardScaler()
X_train_s = scaler_s.fit_transform(X_train_s)
X_test_s = scaler_s.transform(X_test_s)
model_s = SVM_HC()
model_s.fit(X_train_s, y_train_s)
y_pred_s = model_s.predict(X_test_s)
print("Social", accuracy_score(y_test_s, y_pred_s))

df_t = pd.read_csv('titanic.csv')
df_t['Age'] = df_t['Age'].fillna(df_t['Age'].median())
df_t['Embarked'] = df_t['Embarked'].fillna(df_t['Embarked'].mode()[0])
df_t = df_t.drop(columns=['Cabin'])
for col in df_t.columns:
    if df_t[col].dtype == 'object':
        df_t[col] = LabelEncoder().fit_transform(df_t[col])
X_t = df_t.iloc[:, :-1].values
y_t = df_t.iloc[:, -1].values
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_t, y_t, test_size=0.2, random_state=42)
scaler_t = StandardScaler()
X_train_t = scaler_t.fit_transform(X_train_t)
X_test_t = scaler_t.transform(X_test_t)
model_t = SVM_HC()
model_t.fit(X_train_t, y_train_t)
y_pred_t = model_t.predict(X_test_t)
print("Titanic", accuracy_score(y_test_t, y_pred_t))
