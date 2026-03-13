# --------------------------------------------------------------------------------------------
# 11. Classification Techniques: Implement K-Nearest Neighbors (KNN) and Support
#     Vector Machines (SVM) for image classification and evaluate performance.
#
# ANTHONY PINTO ROBINSON (253MCA37)
# 11/03/2026
# --------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the image dataset
digits = load_digits()
X = digits.data      # Features (pixel values)
y = digits.target    # Target labels

print(f"Dataset shape: {X.shape}")
print(f"Target shape: {y.shape}")

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocess the data (Scaling is important for distance-based algorithms like KNN)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Implement and Evaluate KNN
print("\n--- K-Nearest Neighbors (KNN) Classification ---")

# Choose a number of neighbors (k)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

y_pred_knn = knn.predict(x_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(f"KNN Accuracy: {accuracy_knn:.4f}")

# Evaluation metrics
print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

print("\nKNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

# Implement and Evaluate SVM
print("\n--- Support Vector Machine (SVM) Classification ---")

svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(x_train, y_train)

y_pred_svm = svm_model.predict(x_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(f"SVM Accuracy: {accuracy_svm:.4f}")

# Evaluation metrics
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

print("\nSVM Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))