import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

k = int(input("Enter value of K: "))

df_diabetes = pd.read_csv('diabetes_dataset.csv')
print("Diabetes")
print(df_diabetes.head())
print("Missing values")
print(df_diabetes.isnull().sum())

df_social = pd.read_csv('Social_Network_Ads.csv')
print("\nSocial Network Ads")
print(df_social.head())
print("Missing values")
print(df_social.isnull().sum())

df_titanic = pd.read_csv('titanic.csv')
print("\nTitanic")
print(df_titanic.head())
print("Missing values")
print(df_titanic.isnull().sum())

df_titanic['Age'] = df_titanic['Age'].fillna(df_titanic['Age'].median())
df_titanic['Embarked'] = df_titanic['Embarked'].fillna(df_titanic['Embarked'].mode()[0])
df_titanic = df_titanic.drop(columns=['Cabin'])

print("Missing values (after analysis)")
print(df_titanic.isnull().sum())

for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    df_diabetes[col] = df_diabetes[col].replace(0, df_diabetes[col].median())

X_d = StandardScaler().fit_transform(df_diabetes.drop('Outcome', axis=1))
y_d = df_diabetes['Outcome'].values
X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(X_d, y_d, test_size=0.2, random_state=42)

df_social['Gender'] = LabelEncoder().fit_transform(df_social['Gender'])
X_s = StandardScaler().fit_transform(df_social.drop(['User ID', 'Purchased'], axis=1))
y_s = df_social['Purchased'].values
X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_s, y_s, test_size=0.2, random_state=42)

df_titanic = df_titanic.drop(columns=['PassengerId', 'Name', 'Ticket'])
df_titanic['Sex'] = LabelEncoder().fit_transform(df_titanic['Sex'])
df_titanic['Embarked'] = LabelEncoder().fit_transform(df_titanic['Embarked'])
X_t = StandardScaler().fit_transform(df_titanic.drop('Survived', axis=1))
y_t = df_titanic['Survived'].values
X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(X_t, y_t, test_size=0.2, random_state=42)

datasets = [
    ("Diabetes", X_tr_d, X_te_d, y_tr_d, y_te_d),
    ("Social Ads", X_tr_s, X_te_s, y_tr_s, y_te_s),
    ("Titanic", X_tr_t, X_te_t, y_tr_t, y_te_t)
]

for name, X_tr, X_te, y_tr, y_te in datasets:
    print(f"\n{name} Results")
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_tr, y_tr)
    acc = model.score(X_te, y_te)
    print(f"Accuracy: {acc:.4f}")
