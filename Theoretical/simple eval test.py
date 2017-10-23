# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# b = 1. c= 2, D=3, T=4
import numpy as np
import sklearn.metrics

y_true = np.array([[1, 0, 0, 0],
                   [1, 0, 0, 0],
                   [1, 0, 0, 0],
                   [1, 0, 0, 0],
                   [0, 0, 1, 1],
                   [0, 0, 1, 1],
                   [0, 0, 1, 1],
                   [0, 0, 1, 1]
                   ])

y_pred = np.array([[1, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 1],
                   [1, 0, 1, 0],
                   [1, 1, 0, 0],
                   [0, 1, 0, 1],
                   [0, 0, 0, 1],
                   [0, 0, 1, 1]
                   ])

print('Subset accuracy(exact match): {0}'.format(
    sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)))
print('F1_per class, combining Precision,recall,F1 and freq: {0}'.format(
    sklearn.metrics.precision_recall_fscore_support(y_true, y_pred)))
print('F1, avarging_no of class: {0}'.format(sklearn.metrics.f1_score(y_true, y_pred, average='macro')))
print('F1, avarging_freq: {0}'.format(sklearn.metrics.(y_true, y_pred, average='weighted')))