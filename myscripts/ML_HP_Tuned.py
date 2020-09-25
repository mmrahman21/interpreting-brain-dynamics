from __future__ import print_function
import random
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import (f1_score, confusion_matrix, roc_auc_score,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  #For logistic regression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import pickle as p
from torch.utils.data import RandomSampler, BatchSampler
from torchvision import transforms
from torch.autograd import Variable
import torchvision.transforms.functional as TF
import torch.nn.utils.rnn as tn
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV


def _scorer(clf, X, y):
    '''Function that scores a classifier according to what is available as a
    predict function.
    Input:
    - clf = Fitted classifier object
    - X = input data matrix
    - y = estimated labels
    Output:
    - AUC score for binary classification or F1 for multiclass
    The order of priority is as follows:
    - predict_proba
    - decision_function
    - predict
    '''
    n_class = len(np.unique(y))
    if n_class == 2:
        if hasattr(clf, 'predict_proba'):
            ypred = clf.predict_proba(X)
            try:
                ypred = ypred[:, 1]
            except:
                print('predict proba return shape{}'.format(ypred.shape))

            assert len(ypred.shape) == 1,\
                'predict proba return shape {}'.format(ypred.shape)
        elif hasattr(clf, 'decision_function'):
            ypred = clf.decision_function(X)
            assert len(ypred.shape) == 1,\
                'decision_function return shape {}'.format(ypred.shape)
        else:
            ypred = clf.predict(X)
            
        print('Calculating ROC score')
        score = roc_auc_score(y, ypred)
    else:
        score = f1_score(y, clf.predict(X), average='weighted')
    return score

# SVM Classifier
class SVMTrainer():
    def __init__(self, tr_labels, test_labels, device, name):
        self.tr_labels = tr_labels
        self.test_labels = test_labels
        self.device = device
        self.name = name

    def train(self, tr_eps, tst_eps, subjects_per_group, trial_no, run_dir):

        if self.name == "LinearSVM":
            svclassifier = LinearSVC(dual= False, class_weight='balanced' )  # For linear SVM
            param_grid = {'C': [0.01, 0.1, 1],
                          'penalty': ['l1', 'l2']}

        else:
            svclassifier = SVC(C=1, probability=True, cache_size=10000,
                               class_weight='balanced')
            param_grid = {'kernel': ['rbf', 'poly'],
                          'C': [0.01, 0.1, 1]}

        scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=svclassifier, param_grid=param_grid,
                                   cv=3, n_jobs=1, scoring=_scorer, verbose=1)


        # Fit the grid search to the data
        grid_search.fit(tr_eps, self.tr_labels)
        print(grid_search.best_params_)

        best_grid = grid_search.best_estimator_

        y_pred = best_grid.predict(tst_eps)
        
        X = tst_eps
        y = self.test_labels
        
        accuracy = accuracy_score(y, y_pred)
        auc = _scorer(best_grid, X , y)
        
        print("Grid Searched {} Accuracy: {}".format(self.name, accuracy))
        print("Grid Searched {} AUC: {}".format(self.name, auc))

        return accuracy, auc

# Decision Tree Classifier
class DTTrainer():
    def __init__(self, tr_labels, test_labels, device):
        self.tr_labels = tr_labels
        self.test_labels = test_labels
        self.device = device

    def train(self, tr_eps, tst_eps, subjects_per_group, trial_no, run_dir):

        # Base Decision Tree
        dtclf = DecisionTreeClassifier(max_depth=None, max_features='auto')

        param_grid = {
            'criterion': ['gini', 'entropy']}

        scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=dtclf, param_grid=param_grid,
                                   cv=3, n_jobs=1, scoring=scoring, refit='AUC', verbose=1)

        if not os.path.exists('{}/hpTuning/DTmodels/'.format(run_dir)):
            os.makedirs('{}/hpTuning/DTmodels/'.format(run_dir))

        # Fit the grid search to the data
        grid_search.fit(tr_eps, self.tr_labels)
        pprint(grid_search.best_params_)

        best_grid = grid_search.best_estimator_

        with open(run_dir + '/hpTuning/DTmodels/' + '/model_' + str(subjects_per_group)+'_trial_'+ str(trial_no) +'.pkl', 'wb') as f:
            p.dump(best_grid, f, protocol=2)
            print('Model Saved')

        y_pred = best_grid.predict(tst_eps)

        predictions = torch.from_numpy(y_pred)
        y_score = best_grid.predict_proba(tst_eps)[:, 1]

        accuracy = calculate_accuracy_by_labels(predictions, self.test_labels.to(self.device))
        print("Grid Searched DT Accuracy:", accuracy)
        fpr, tpr, thresholds = metrics.roc_curve(self.test_labels, y_score, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print("Grid Searched DT AUC: {}".format(auc))

        return accuracy, auc


# Random Forests Classifier
class RFTrainer():
    def __init__(self, tr_labels, test_labels, device):
        self.tr_labels = tr_labels
        self.test_labels = test_labels
        self.device = device

    def train(self, tr_eps, tst_eps, subjects_per_group, trial_no, run_dir):

        # Base Random Forest
        rfclf = RandomForestClassifier(max_depth=None, n_estimators=10, max_features='auto')

        param_grid = {
            'criterion' : ['gini', 'entropy'],
            'n_estimators': list(range(5, 20))}

        scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=rfclf, param_grid=param_grid,
                                   cv=3, n_jobs=1, scoring=scoring, refit='AUC', verbose=1)

        if not os.path.exists('{}/hpTuning/RFmodels/'.format(run_dir)):
            os.makedirs('{}/hpTuning/RFmodels/'.format(run_dir))

        # Fit the grid search to the data
        grid_search.fit(tr_eps, self.tr_labels)
        pprint(grid_search.best_params_)

        best_grid = grid_search.best_estimator_

        with open(run_dir+'/hpTuning/RFmodels/'+ '/model_'+str(subjects_per_group)+'_trial_'+ str(trial_no) +'.pkl', 'wb') as f:
            p.dump(best_grid, f, protocol=2)
            print('Model Saved')

        y_pred = best_grid.predict(tst_eps)

        predictions = torch.from_numpy(y_pred)
        y_score = best_grid.predict_proba(tst_eps)[:, 1]

        accuracy = calculate_accuracy_by_labels(predictions, self.test_labels.to(self.device))
        print("Grid Searched RF Accuracy:", accuracy)
        fpr, tpr, thresholds = metrics.roc_curve(self.test_labels, y_score, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        print("Grid Searched RF AUC: {}".format(auc))

        return accuracy, auc


# Logistic Regression Classifier
class LRTrainer():
    def __init__(self, tr_labels, test_labels, device):
        self.tr_labels = tr_labels
        self.test_labels = test_labels
        self.device = device

    def train(self, tr_eps, tst_eps, subjects_per_group, trial_no, run_dir):

        lrclf = LogisticRegression(fit_intercept=True, solver='lbfgs',
                                   penalty='l2')

        param_grid = {'C': [0.001, 0.1, 1]}

        scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=lrclf, param_grid=param_grid,
                                   cv=3, n_jobs=1, scoring=_scorer, verbose=1)


        # Fit the grid search to the data
        grid_search.fit(tr_eps, self.tr_labels)
        pprint(grid_search.best_params_)

        best_grid = grid_search.best_estimator_
        y_pred = best_grid.predict(tst_eps)
        
        X = tst_eps
        y = self.test_labels
        
        accuracy = accuracy_score(y, y_pred)
        auc = _scorer(best_grid, X , y)


        print("Grid Searched LR Accuracy:", accuracy)
        print("Grid Searched LR AUC: {}".format(auc))

        return accuracy, auc


# Naive Bayes Classifier
class NBTrainer():
    def __init__(self, tr_labels, test_labels, device):
        self.tr_labels = tr_labels
        self.test_labels = test_labels
        self.device = device

    def train(self, tr_eps, tst_eps, subjects_per_group, trial_no, run_dir):
        # Create a Gaussian Classifier
        nbclf = GaussianNB()

        param_grid = {}

        scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=nbclf, param_grid=param_grid,
                                   cv=3, n_jobs=1, scoring=scoring, refit='AUC', verbose=1)

        if not os.path.exists('{}/hpTuning/NBmodels/'.format(run_dir)):
            os.makedirs('{}/hpTuning/NBmodels/'.format(run_dir))

        # Fit the grid search to the data
        grid_search.fit(tr_eps, self.tr_labels)
        pprint(grid_search.best_params_)

        best_grid = grid_search.best_estimator_

        with open(run_dir + '/hpTuning/NBmodels/' + '/model_' + str(subjects_per_group)+'_trial_'+ str(trial_no)  + '.pkl', 'wb') as f:
            p.dump(best_grid, f, protocol=2)
            print('Model Saved')

        y_pred = best_grid.predict(tst_eps)

        predictions = torch.from_numpy(y_pred)
        y_score = best_grid.predict_proba(tst_eps)[:, 1]

        accuracy = calculate_accuracy_by_labels(predictions, self.test_labels.to(self.device))
        print("Grid Searched NB Accuracy:", accuracy)
        fpr, tpr, thresholds = metrics.roc_curve(self.test_labels, y_score, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print("Grid Searched NB AUC: {}".format(auc))

        return accuracy, auc

# KNN Classifier
class KNNClassifier():
    def __init__(self, tr_labels, test_labels, device):
        self.tr_labels = tr_labels
        self.test_labels = test_labels
        self.device = device

    def train(self, tr_eps, tst_eps, subjects_per_group, trial_no, run_dir):

        # Creating classifier object
        knnclf = KNeighborsClassifier()

        param_grid = {'n_neighbors': [1, 5, 10, 20]}

        scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=knnclf, param_grid=param_grid,
                                   cv=3, n_jobs=1, scoring=scoring, refit='AUC', verbose=1)

        if not os.path.exists('{}/hpTuning/KNNmodels/'.format(run_dir)):
            os.makedirs('{}/hpTuning/KNNmodels/'.format(run_dir))

        # Fit the grid search to the data
        grid_search.fit(tr_eps, self.tr_labels)
        pprint(grid_search.best_params_)

        best_grid = grid_search.best_estimator_

        with open(run_dir + '/hpTuning/KNNmodels/' + '/model_' + str(subjects_per_group) +'_trial_'+ str(trial_no) + '.pkl', 'wb') as f:
            p.dump(best_grid, f, protocol=2)
            print('Model Saved')

        y_pred = best_grid.predict(tst_eps)

        predictions = torch.from_numpy(y_pred)
        y_score = best_grid.predict_proba(tst_eps)[:, 1]

        accuracy = calculate_accuracy_by_labels(predictions, self.test_labels.to(self.device))
        print("Grid Searched KNN Accuracy:", accuracy)
        fpr, tpr, thresholds = metrics.roc_curve(self.test_labels, y_score, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print("Grid Searched KNN AUC: {}".format(auc))

        return accuracy, auc

class FCNetwork():
    def __init__(self, tr_labels, test_labels, device):
        self.tr_labels = tr_labels
        self.test_labels = test_labels
        self.device = device

    def train(self, tr_eps, tst_eps, subjects_per_group, trial_no, run_dir):

        # Creating classifier object
        mlpclf = MLP()

        param_grid = {
            'hidden_layer_sizes': [(100, 50), (50, 25)],
            'max_iter': [500, 1000]}
        scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=mlpclf, param_grid=param_grid,
                                   cv=3, n_jobs=1, scoring=_scorer, verbose=1)

        # Fit the grid search to the data
        grid_search.fit(tr_eps, self.tr_labels)
        pprint(grid_search.best_params_)

        best_grid = grid_search.best_estimator_
        y_pred = best_grid.predict(tst_eps)
        
        X = tst_eps
        y = self.test_labels
        
        accuracy = accuracy_score(y, y_pred)
        auc = _scorer(best_grid, X , y)
        
        print("Grid Searched MLP Accuracy:", accuracy)
        print("Grid Searched MLP AUC: {}".format(auc))

        return accuracy, auc
