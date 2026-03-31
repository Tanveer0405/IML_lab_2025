import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean, cityblock

k = int(input("Enter value of K: "))

df_d = pd.read_csv('diabetes_dataset.csv')
print("Diabetes")
print(df_d.head())
print("Missing values")
print(df_d.isnull().sum())

df_s = pd.read_csv('Social_Network_Ads.csv')
print("\nSocial Network Ads")
print(df_s.head())
print("Missing values")
print(df_s.isnull().sum())

df_t = pd.read_csv('titanic.csv')
print("\nTitanic")
print(df_t.head())
print("Missing values")
print(df_t.isnull().sum())

df_t['Age'] = df_t['Age'].fillna(df_t['Age'].median())
df_t['Embarked'] = df_t['Embarked'].fillna(df_t['Embarked'].mode()[0])
df_t = df_t.drop(columns=['Cabin'])

print("Missing values.")
print(df_t.isnull().sum())

def dtw_dist(s, t):
    n, m = len(s), len(t)
    dtw = np.zeros((n + 1, m + 1))
    dtw[1:, 0] = np.inf
    dtw[0, 1:] = np.inf
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = abs(s[i-1] - t[j-1])
            dtw[i, j] = c + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    return dtw[n, m]

def knn_pred(X_tr, y_tr, X_te, k, m):
    preds = []
    for tp in X_te:
        dists = []
        for tr in X_tr:
            if m == 'L1':
                d = cityblock(tp, tr)
            elif m == 'L2':
                d = euclidean(tp, tr)
            elif m == 'DTW':
                d = dtw_dist(tp, tr)
            dists.append(d)
        idx = np.argsort(dists)[:k]
        preds.append(np.bincount(y_tr[idx]).argmax())
    return np.array(preds)

for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    df_d[col] = df_d[col].replace(0, df_d[col].median())

X_d = StandardScaler().fit_transform(df_d.drop('Outcome', axis=1))
y_d = df_d['Outcome'].values
X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(X_d, y_d, test_size=0.2, random_state=42)

df_s['Gender'] = LabelEncoder().fit_transform(df_s['Gender'])
X_s = StandardScaler().fit_transform(df_s.drop(['User ID', 'Purchased'], axis=1))
y_s = df_s['Purchased'].values
X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_s, y_s, test_size=0.2, random_state=42)

df_t = df_t.drop(columns=['PassengerId', 'Name', 'Ticket'])
df_t['Sex'] = LabelEncoder().fit_transform(df_t['Sex'])
df_t['Embarked'] = LabelEncoder().fit_transform(df_t['Embarked'])
X_t = StandardScaler().fit_transform(df_t.drop('Survived', axis=1))
y_t = df_t['Survived'].values
X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(X_t, y_t, test_size=0.2, random_state=42)

data = [
    ("Diabetes", X_tr_d, X_te_d, y_tr_d, y_te_d),
    ("Social Ads", X_tr_s, X_te_s, y_tr_s, y_te_s),
    ("Titanic", X_tr_t, X_te_t, y_tr_t, y_te_t)
]

for name, X_tr, X_te, y_tr, y_te in data:
    print(f"\n{name} Results")
    for m in ['L1', 'L2', 'DTW']:
        y_p = knn_pred(X_tr, y_tr, X_te, k, m)
        acc = (y_p == y_te).mean()
        print(f"{m} Accuracy: {acc:.4f}")