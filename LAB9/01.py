import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, silhouette_score
from scipy.stats import mode

df_d = pd.read_csv('diabetes_dataset.csv')
df_s = pd.read_csv('Social_Network_Ads.csv')
df_t = pd.read_csv('titanic.csv')

df_d.fillna(df_d.mean(numeric_only=True), inplace=True)
df_s.fillna(df_s.mean(numeric_only=True), inplace=True)

df_t['Age'] = df_t['Age'].fillna(df_t['Age'].median())
df_t['Embarked'] = df_t['Embarked'].fillna(df_t['Embarked'].mode()[0])
df_t = df_t.drop(columns=['Cabin'])

datasets = {
    "Diabetes Dataset": df_d,
    "Social Dataset": df_s,
    "Titanic Dataset": df_t
}

for name, df in datasets.items():
    print(f"\n{name}")    
    if name == "Diabetes Dataset":
        y_true = df['Outcome'].values
        X = df.drop(columns=['Outcome'])
    elif name == "Social Dataset":
        y_true = df['Purchased'].values
        X = df.drop(columns=['Purchased'])
    elif name == "Titanic Dataset":
        y_true = df['Survived'].values
        X = df.drop(columns=['Survived'])

    
    X = X.select_dtypes(include=[np.number]).values   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)   

    print("\nCluster Centers:\n", kmeans.cluster_centers_)
    print("\nCluster Counts:")
    print(pd.Series(labels).value_counts())
    print("\nInertia :", kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, labels)
    print("Silhouette Score:", sil_score)
    
    labels_reshaped = labels.reshape(-1, 1)
    mapped_labels = np.zeros_like(labels_reshaped)
    for i in range(2):
        mask = (labels_reshaped == i)
        if np.sum(mask) == 0:
            continue
        mapped_labels[mask] = mode(y_true[mask.flatten()], keepdims=True)[0]
    acc = accuracy_score(y_true, mapped_labels)
    print("Accuracy:", round(acc, 4))
