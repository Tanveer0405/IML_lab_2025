# Create State of art table for all the algorithms used in the lab

import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

datasets = {
    "Diabetes Dataset": df_d,
    "Social Dataset": df_s,
    "Titanic Dataset": df_t
}
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel='rbf')
}

results = []

for dataset_name, df in datasets.items():
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results.append([
            model_name,
            dataset_name,
            round(acc, 2),
            round(prec, 2),
            round(rec, 2),
            round(f1, 2)
        ])

final_df = pd.DataFrame(results, columns=[
    "Model / Method",
    "Dataset",
    "Accuracy (%)",
    "Precision",
    "Recall",
    "F1 Score"
])


print("\nTABLE\n")
print(final_df)




