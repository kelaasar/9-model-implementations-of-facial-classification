import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


dataset = pd.read_csv("facial_expressions/data/legend.csv")
dataset = dataset.drop(["user.id"], axis=1)
dataset['emotion'] = dataset['emotion'].str.lower()

class_mapping = dict( zip( dataset["emotion"].astype('category').cat.codes, dataset["emotion"]))
dataset["emotion_class"] = dataset["emotion"].astype('category').cat.codes

xtrain, xval, ytrain, yval = train_test_split(dataset["image"], dataset["emotion_class"], test_size=0.2, random_state=77, stratify=dataset["emotion_class"])

xtraindata = np.zeros((len(xtrain), 128, 128, 3))
xvaldata = np.zeros((len(xval), 128, 128, 3))
wierd_list = []
for i in range(len(xtrain)):
    xtraindata[i] = cv2.resize(cv2.imread("facial_expressions/images/" + xtrain.iloc[i]), (128,128)) / 255.          # resizing due to colab RAM constraints

for i in range(len(xval)):
    xvaldata[i] = cv2.resize(cv2.imread("facial_expressions/images/" + xval.iloc[i]), (128,128)) / 255.

# Flattened image data for non-spatial models
xtraindata_flat = xtraindata.reshape(len(xtraindata), -1)
xvaldata_flat = xvaldata.reshape(len(xvaldata), -1)

LR = LogisticRegression()
LRparam_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2'],
    'max_iter': [100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}
LR_search = GridSearchCV(LR, param_grid=LRparam_grid, refit = True, verbose = 3, cv=5, n_jobs=4)

LR_search.fit(xtraindata_flat , ytrain)
LR_search.best_params_

print('Mean Accuracy: %.3f' % LR_search.best_score_)
print('Config: %s' % LR_search.best_params_)

logreg = LR_search.best_estimator_
y_pred = logreg.predict(xvaldata_flat)

accuracy = accuracy_score(yval, y_pred)
print(f"Accuracy: {accuracy}")