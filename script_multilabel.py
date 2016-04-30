# Authors: Vlad Niculae, Mathieu Blondel
# License: BSD 3 clause
"""
=========================
Multilabel classification
=========================

This example simulates a multi-label document classification problem. The
dataset is generated randomly based on the following process:

    - pick the number of labels: n ~ Poisson(n_labels)
    - n times, choose a class c: c ~ Multinomial(theta)
    - pick the document length: k ~ Poisson(length)
    - k times, choose a word: w ~ Multinomial(theta_c)

In the above process, rejection sampling is used to make sure that n is more
than 2, and that the document length is never zero. Likewise, we reject classes
which have already been chosen.  The documents that are assigned to both
classes are plotted surrounded by two colored circles.

The classification is performed by projecting to the first two principal
components found by PCA and CCA for visualisation purposes, followed by using
the :class:`sklearn.multiclass.OneVsRestClassifier` metaclassifier using two
SVCs with linear kernels to learn a discriminative model for each class.
Note that PCA is used to perform an unsupervised dimensionality reduction,
while CCA is used to perform a supervised one.

Note: in the plot, "unlabeled samples" does not mean that we don't know the
labels (as in semi-supervised learning) but that the samples simply do *not*
have a label.
"""

import numpy as np
import random

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn import cross_validation, linear_model
from sklearn.metrics import f1_score

import sys
import pdb
import os

def load_lbl(lbl_fn):

    lbl_list = []
    with open(lbl_fn,'r') as fid:
        for aline in fid:
            parts = aline.strip().split()
            lbl_list.append((int(parts[0]),int(parts[1])))

    return lbl_list

def load_fea(fea_fn):
    num_ins = 0
    fea_num = 0
    with open(fea_fn) as fid:
        for aline in fid:
            num_ins += 1
            fea_len = len(aline.strip().split())
            if fea_num == 0:
                fea_num = fea_len
            else:
                assert(fea_len ==  fea_num)

    np_fea = np.zeros((num_ins, fea_num -1 ))
    with open(fea_fn) as fid:
        for idx, aline in enumerate(fid):
            fea = [ float(f) for f in aline.strip().split()[1:] ]
            np_fea[idx,:] = fea
    return np_fea
if __name__ == '__main__':
   
    if len(sys.argv) < 2:
        print 'Usage: {0} <lbl_fn> <fea_fn>'.format(sys.argv[0])
        sys.exit()
    
    lbl = load_lbl(sys.argv[1])
    fea = load_fea(sys.argv[2])
    lr = linear_model.LogisticRegression()
    
    percentage = np.arange(0.1,1,0.1)

    classif = OneVsRestClassifier(lr)
    for p in percentage:
        random.shuffle(lbl)
        train_ins = int(len(lbl) * p)
        test_ins = lbl[train_ins:]
        train_ins = lbl[0:train_ins]

        X = np.zeros((len(train_ins), fea.shape[1]))
        Y = np.zeros((len(train_ins)))

        X_test = np.zeros((len(test_ins), fea.shape[1]))
        Y_test = np.zeros((len(test_ins)))

        for idx, tup in enumerate(train_ins):
            X[idx,:] = fea[tup[0],:]
            Y[idx] = tup[1]

        for idx, tup in enumerate(test_ins):
            X_test[idx,:] = fea[tup[0],:]
            Y_test[idx] = tup[1]


        classif.fit(X, Y)
        Y_pred = classif.predict(X_test)
        f1_a = f1_score(Y_test, Y_pred, average='macro')
        f1_i = f1_score(Y_test, Y_pred, average='micro')
        print 'Macro', f1_a
        print 'Micro', f1_i



