import pickle

import pandas as pd
import numpy as np
import rrcf
import dill
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataset_sdn.csv')
df=df.head(10000)
X = df[['pktcount', 'bytecount']]
y = df['label']
print(X.head())
X_norm = (X - X.mean()) / X.std()
X_norm = X_norm.dropna()
# Create train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

num_trees = 1000
forest_normal = []
forest_attack = []

with open('forest_normal.pkl', 'rb') as file:
    forest_normal = pickle.load(file)
with open('forest_attack.pkl', 'rb') as file:
    forest_attack = pickle.load(file)

# Compute anomaly score for each new point
n=8000
avg_codisp = np.zeros((len(X_test), 2))
X_test=X_test.to_numpy()
for index in range(len(X_test)):
    for tree_normal, tree_attack in zip(forest_normal, forest_attack):
        tree_normal.insert_point(X_test[index], index=index+n)
        tree_attack.insert_point(X_test[index], index=index+n)
        avg_codisp[index, 0] += tree_normal.codisp(index+n)
        avg_codisp[index, 1] += tree_attack.codisp(index+n)
        tree_normal.forget_point(index+n)
        tree_attack.forget_point(index+n)

avg_codisp /= num_trees

# Predict labels for test dataset
predictions = np.argmin(avg_codisp, axis=1)
print(predictions)
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

accuracy=accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)