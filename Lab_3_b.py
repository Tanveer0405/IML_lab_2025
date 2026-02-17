import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_curve, auc, accuracy_score, f1_score

df = pd.read_csv("daily-minimum-temperatures-in-me.csv")

df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
df["Daily minimum temperatures"] = pd.to_numeric(df["Daily minimum temperatures"], errors='coerce')
df = df.dropna()

df["DistFromWinter"] = df["Date"].apply(lambda x: abs(x.dayofyear - 196))

X = df[["DistFromWinter"]]
y = df["Daily minimum temperatures"]

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
print(f"R2 Score: {r2:.4f}")

threshold = y.mean()
y_true_binary = (y >= threshold).astype(int)

y_pred_binary = (y_pred >= threshold).astype(int)

accuracy = accuracy_score(y_true_binary, y_pred_binary)
f1 = f1_score(y_true_binary, y_pred_binary)

print(f"Accuracy: {accuracy:.4f}")


fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred)
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', s=1, alpha=0.3)
plt.plot(X, y_pred, color='red', linewidth=2)
plt.title(f"Linear Regression (R2: {r2:.2f})")
plt.xlabel("Distance from Winter (Days)")
plt.ylabel("Temperature")

plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve ({roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()
