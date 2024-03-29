import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import warnings
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Ignore all warnings
warnings.filterwarnings("ignore")

# Read the csv data
InputDataFrame = pd.read_csv('data.csv')


# 1. Extract features values and Image Ids from the data:
# In this 2 class classification problem Make B = 0 and M = 1
def partition(x):
    if x == 'B':
        return 0
    return 1


actualClass = InputDataFrame['diagnosis']
DiagnosisClass = actualClass.map(partition)
InputDataFrame['diagnosis'] = DiagnosisClass
# print("Number of data points in our data", InputDataFrame.shape)
# print(InputDataFrame.head(5))

# Now, we will be splitting the following data into labels and features.
DiagnosisList = InputDataFrame['diagnosis']
FinalDataFrame = InputDataFrame.drop('diagnosis', axis=1)
# print(FinalDataFrame.head(5))
# Handling Inf and NaN values
FinalDataFrame[:] = np.nan_to_num(FinalDataFrame)

# 2.Data Partitioning - 80% for Training and remaining for validation and testing
X = FinalDataFrame.values
Y = DiagnosisList

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# X_train, X_cv, Y_train, Y_cv = train_test_split(X_train,Y_train,test_size=0.2)
# print("Train :",X_train.shape,Y_train.shape)
# print("CV:",X_cv.shape,Y_cv.shape)
# print("Test:",X_test.shape,Y_test.shape)

# 3. Train using Logistic Regression
# Use Gradient Descent for logistic regression to train the model using a group of hyperparameters.

# 4. Tune hyper-parameters: Validate the regression performance of your model on the validation
logistic = LogisticRegression()
penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)
hyperparameters = dict(C=C, penalty=penalty)

# Create grid search using 5-fold cross validation
clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
# Fit grid search
best_model = clf.fit(X_train, Y_train)
# Best Hyperparameters
# print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
# print('Best C:', best_model.best_estimator_.get_params()['C'])
C = best_model.best_estimator_.get_params()['C']
penalty = best_model.best_estimator_.get_params()['penalty']

# 5. Test your machine learning scheme on the testing set:
model = LogisticRegression(C=C, penalty=penalty)
model.fit(X_train, Y_train)
Y_predicted = model.predict(X_test)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y_test, Y_predicted)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(Y_test, Y_predicted)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(Y_test, Y_predicted)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(Y_test, Y_predicted)
print('F1 score: %f' % f1)
