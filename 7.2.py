import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('suv.csv')

# feature variable X ['Age','EstimatedSalary']
X = df[['Age','EstimatedSalary']]

# target variable y ['Purchased']
y = df['Purchased']

# split dataset into 80-20 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Scale feature using StandardScaler
scaler  = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train model using DecisionTreeClassifier with entropy 
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=5)
dt_entropy.fit(X_train, y_train)
y_pred_entropy = dt_entropy.predict(X_test)


# Confusion matrix and classification report
dt_entropy_conf = confusion_matrix(y_test, y_pred_entropy)
print(f'Confusion Matrix Entropy: {dt_entropy_conf}')

dt_entropy_class = classification_report(y_test, y_pred_entropy)
print(f'Classification Report Entropy: {dt_entropy_class}')



# Again train model using DecisionTreeClassifier with gini 
dt_gini = DecisionTreeClassifier(criterion='gini')
dt_gini.fit(X_train, y_train)
y_pred_gini = dt_gini.predict(X_test)

# Confusion matrix and classification report gini
dt_gini_conf = confusion_matrix(y_test, y_pred_gini)
print(f'Confusion Matrix Gini: {dt_gini_conf}')

dt_gini_class = classification_report(y_test, y_pred_gini)
print(f'Classification Report Gini: {dt_gini_class}')


'''
Here, entropy produced 6 false positives, and 6 false negatives. Whereas, gini produced 6 false positives and 4 false negatives. From this it is seen that gini is better at identifying class 1.
In case of class 0, both the models have performed identically producing 6 false positives.

Overall gini has performed better as shown by 88% accuracy compared to that of Entropy's 85%.

Gini is more precise for negative (0) class and for class 1, it is slightly better than entropy (0.79 vs 0.78)
'''