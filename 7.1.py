import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report


df = pd.read_csv('data_banknote_authentication.csv')

# Selecting feature and target variables as X and y
X = df.drop(columns=['class'])
y = df['class']

# split dataset into 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)


# Train SVM Model Linear
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# compute confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix (Kernel=Linear): {conf_matrix}')

class_report = classification_report(y_test, y_pred)
print(f'Classification Report (Kernel=Linear): {class_report}')


# Train SVM model RBF
svm_model_rbf = SVC(kernel='rbf')
svm_model_rbf.fit(X_train, y_train)
y_pred_rbf = svm_model_rbf.predict(X_test)

# compute confusion matrix and classification report for rbf kernel
conf_matrix_rbf = confusion_matrix(y_test, y_pred_rbf)
print(f'Confusion Matrix (Kernel=RBF): {conf_matrix_rbf}')

class_report_rbf = classification_report(y_test, y_pred_rbf)
print(f'Classification Report (Kernel=RBF): {class_report_rbf}')



'''

Based on the results, both models performed excceptionally well. However, there are tiny differences:
- Linear karnel produced 2 false positives resulting in 99% accuracy. Precision, recall, and f1-score are very high but not perfect.
- RBF didn't misclassified any data; meaning there's no false results. Hence, the accuracy is 100%. recision, recall, and f1-score all are perfect.

Linear kernel assumes that the data is linearly separable. It performed very well, but a small number of data were incorrectly classified (2 FP).

RBF kernel is more flexible and can model complex, nonlinear decision boundries. In the above case, it perfectly classified all the data (no false results); meaning, it outperformed the linear model.

Since RBF model generated 100% accuracy, it suggests that a nonlinear decision boundry is optimal for this dataset.


'''