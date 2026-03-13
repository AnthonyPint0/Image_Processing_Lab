# # ----------------------------------------------------------------------------------------------
# 12. Bayesian Classifier and Performance Evaluation: Implement a Bayesian classifier 
#       and evaluate using accuracy, precision, recall, F1-score, and confusion matrix.

#      ANTHONY PINTO ROBINSON (253MCA37)
#      11/03/2026
# --------------------------------------------------------------------------------------------

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
X, y = load_iris(return_X_y=True)
target_names = load_iris().target_names

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Training
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predictions
y_pred = gnb.predict(X_test)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=target_names))

print("\nConfusion Matrix:\n",
      confusion_matrix(y_test, y_pred))

# Visualization
ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred),
    display_labels=target_names
).plot(cmap=plt.cm.Blues)

plt.show()