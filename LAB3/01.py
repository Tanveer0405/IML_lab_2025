import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

df = pd.read_csv("daily-minimum-temperatures-in-me.csv")

df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Daily minimum temperatures"] = pd.to_numeric(df["Daily minimum temperatures"], errors="coerce")
df = df.dropna()

df["DistFromWinter"] = df["Date"].apply(lambda x: abs(x.dayofyear - 196))

X = df["DistFromWinter"].values
y = df["Daily minimum temperatures"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

class MeraLR:
    def __init__(self):
        self.m = None
        self.b = None
        
    def fit(self, X_train, y_train):
        x_mean = np.mean(X_train)
        y_mean = np.mean(y_train)
        num = np.sum((X_train - x_mean) * (y_train - y_mean))
        den = np.sum((X_train - x_mean) ** 2)
        self.m = num / den
        self.b = y_mean - self.m * x_mean
        
    def predict(self, X_test):
        return self.m * X_test + self.b

model = MeraLR()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_mean = np.mean(y_test)
ss_total = np.sum((y_test - y_mean) ** 2)
ss_residual = np.sum((y_test - y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)

threshold = y.mean()
y_true_binary = (y_test >= threshold).astype(int)
y_pred_binary = (y_pred >= threshold).astype(int)

TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
TN = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
FP = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
FN = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

accuracy = (TP + TN) / (TP + TN + FP + FN)


fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred)
roc_auc = auc(fpr, tpr)

print("Slope (m):", model.m)
print("Intercept (b):", model.b)
print("R2 Score:", round(r2, 4))
print("Accuracy:", round(accuracy, 4))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, s=1, alpha=0.3)
plt.plot(X_test, y_pred, linewidth=2)
plt.title(f"(HC) Linear Regression (R2: {r2:.2f})")
plt.xlabel("Distance from Winter")
plt.ylabel("Temperature")

plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()
plt.show()
