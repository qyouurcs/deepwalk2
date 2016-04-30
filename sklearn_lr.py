#!/usr/bin/python

import numpy as np
from sklearn import linear_model
from sklearn import datasets
from sklearn.svm import l1_min_c
#from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation, linear_model


label_fn = 'adj_sa.txt'
fea_fn = 'adj_glove.txt'

dict_lbl = {}

with open(label_fn,'r') as fid:
    for aline in fid:
        aline = aline.strip().split()
        if float(aline[1]) > 0:
            dict_lbl[aline[0]] = 1
        else:
            dict_lbl[aline[0]] = 0

dict_fea = {}
fea_len = 0
with open(fea_fn,'r') as fid:
    for aline in fid:
        aline = aline.strip().split()
        dict_fea[aline[0]] = [ float(a) for a in aline[1:] ]
        fea_len = len(aline) - 1

x = np.zeros((len(dict_fea), fea_len))
y = np.zeros((len(dict_fea)))

i=0
for key in dict_fea:
    x[i,:] = dict_fea[key]
    y[i] = dict_lbl[key]
    i+=1

lasso = linear_model.Lasso()

lr = linear_model.LogisticRegression

alphas = np.logspace(-4, -.5, 30)

scores = list()
scores_std = list()

lr = linear_model.LogisticRegression()

this_scores = cross_validation.cross_val_score(lr, x,y, cv = 5, n_jobs = 1)
print('Accuracy: %0.2f (+/- %0.2f)' % (this_scores.mean(), this_scores.std() * 2))
for alpha in alphas:
    lasso.alpha = alpha
    this_scores = cross_validation.cross_val_score(lasso, x,y, cv = 5, n_jobs = 1)
    print("alpha = %f, Accuracy: %0.2f (+/- %0.2f)" % (alpha, this_scores.mean(), this_scores.std() * 2))
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))
