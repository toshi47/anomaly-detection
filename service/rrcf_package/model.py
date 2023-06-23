import pickle
import numpy as np
import pandas as pd
import rrcf
import dill
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split


class Model_RRCF():
    def __init__(self, num_trees):
        self.num_trees = num_trees
        self.forest_normal = []
        self.forest_attack = []
        self.avg_codisp = 0
        self.rrcf = rrcf

    def train(self, X_train, y_train):
        for _ in range(self.num_trees):
            tree_normal = self.rrcf.RCTree(X_train[y_train == 0].values)
            tree_attack = self.rrcf.RCTree(X_train[y_train == 1].values)
            self.forest_normal.append(tree_normal)
            self.forest_attack.append(tree_attack)

    def test(self, X_test, num):
        n = num
        self.avg_codisp = np.zeros((len(X_test), 2))
        X_test = X_test.to_numpy()
        for index in range(len(X_test)):
            for tree_normal, tree_attack in zip(self.forest_normal, self.forest_attack):
                tree_normal.insert_point(X_test[index], index=index + n)
                tree_attack.insert_point(X_test[index], index=index + n)
                self.avg_codisp[index, 0] += tree_normal.codisp(index + n)
                self.avg_codisp[index, 1] += tree_attack.codisp(index + n)
                tree_normal.forget_point(index + n)
                tree_attack.forget_point(index + n)

        self.avg_codisp /= self.num_trees
    def detection_ddos(self, X_test):
        n = 8000
        self.avg_codisp = np.zeros((len(X_test), 2))
        X_test = X_test.to_numpy()
        for index in range(len(X_test)):
            for tree_normal, tree_attack in zip(self.forest_normal, self.forest_attack):
                tree_normal.insert_point(X_test[index], index=index + n)
                tree_attack.insert_point(X_test[index], index=index + n)
                self.avg_codisp[index, 0] += tree_normal.codisp(index + n)
                self.avg_codisp[index, 1] += tree_attack.codisp(index + n)
                tree_normal.forget_point(index + n)
                tree_attack.forget_point(index + n)

        self.avg_codisp /= self.num_trees
        # Predict labels for test dataset
        predictions = np.argmin(self.avg_codisp, axis=1)
        return predictions
    def save_forest(self, name_f_normal, name_f_attack):
        # Save forest_normal to a file
        with open(name_f_normal, 'wb') as file:
            dill.dump(self.forest_normal, file)
        # Save forest_attack to a file
        with open(name_f_attack, 'wb') as file:
            dill.dump(self.forest_attack, file)

    def load_forest(self, name_f_normal, name_f_attack):
        with open(name_f_normal, 'rb') as file:
            self.forest_normal = pickle.load(file)
        with open(name_f_attack, 'rb') as file:
            self.forest_attack = pickle.load(file)

    def print_metrics(self, y_test):
        # Predict labels for test dataset
        predictions = np.argmin(self.avg_codisp, axis=1)
        print('\tMetrics:\t')
        print(f'Accuracy: {accuracy_score(y_test, predictions)}')
        print(f'Precision: {precision_score(y_test, predictions)}')
        print(f'Recall: {recall_score(y_test, predictions)}')
        print(f'f1-score: {f1_score(y_test, predictions)}')
