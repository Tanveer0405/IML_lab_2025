# WAP to find out i) Precision, ii) Recall, iii) F1, iv) Accuracy, v) False Positive, vi) False Negative from your two 1-D datasets ( Original and Predicted ).
# Also plot a confusion matrix of the dataset.

import matplotlib.pyplot as plt

actual = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0]
predicted = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]

n=len(actual)

TP=0
TN=0
FP=0
FN=0
for i in range(n):
    if actual[i] == 1 and predicted[i] == 1:
        TP = TP + 1
    elif actual[i] == 0 and predicted[i] == 1:
        FP = FP + 1
    elif actual[i] == 0 and predicted[i] == 0:
        TN = TN + 1
    elif actual[i] == 1 and predicted[i] == 0:
        FN = FN + 1

# Precision
if (TP + FP) != 0:
    precision = TP / (TP + FP)
else:
    precision = 0

# Recall
if (TP + FN) != 0:
    recall = TP / (TP + FN)
else:
    recall = 0

# F1
if (precision + recall) != 0:
    f1 = 2 * (precision * recall) / (precision + recall)

# Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)


print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"False Positive: {FP:.2f}")
print(f"False Negative: {FN:.2f}")

# Confusion Matrix
confusion_matrix = [
    [TN, FP],
    [FN, TP]
]


plt.imshow(confusion_matrix)
plt.colorbar()

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks([0, 1], ["0", "1"])
plt.yticks([0, 1], ["0", "1"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, confusion_matrix[i][j],
                 ha="center", va="center", color="black")

plt.title("Confusion Matrix")
plt.show()

