# SVM

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df_d = pd.read_csv('diabetes_dataset.csv')
print("Diabetes")
print(df_d.head())
print("Missing values")
print(df_d.isnull().sum())
df_d.fillna(df_d.mean(numeric_only=True), inplace=True)

df_s = pd.read_csv('Social_Network_Ads.csv')
print("\nSocial Network Ads")
print(df_s.head())
print("Missing values")
print(df_s.isnull().sum())
df_s.fillna(df_s.mean(numeric_only=True), inplace=True)

df_t = pd.read_csv('titanic.csv')
print("\nTitanic")
print(df_t.head())
print("Missing values")
print(df_t.isnull().sum())
df_t['Age'] = df_t['Age'].fillna(df_t['Age'].median())
df_t['Embarked'] = df_t['Embarked'].fillna(df_t['Embarked'].mode()[0])
df_t = df_t.drop(columns=['Cabin'])
print("Missing values (after analysis)")
print(df_t.isnull().sum())

datasets={
    "Diabetes": df_d,
    "Social": df_s,
    "Titanic": df_t
}

for name, df in datasets.items():
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col])
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(name, accuracy_score(y_test, y_pred))