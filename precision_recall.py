import pandas as pd
from nltk.corpus import stopwords
import nltk
import re
import os
import random
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import numpy as np

#load model
model = Doc2Vec.load(os.path.join("trained", "comments2vec.d2v"))

with open ('x_train', 'rb') as fp:
	x_train = pickle.load(fp)

with open ('x_test', 'rb') as fp:
	x_test = pickle.load(fp)

with open ('y_train', 'rb') as fp:
	y_train = pickle.load(fp)

with open ('y_test', 'rb') as fp:
	y_test = pickle.load(fp)

y_true = []
for i in (y_test):
	if(i == "REAL"):
		y_true.append(0)
	else:
		y_true.append(1)

y_true_nd = np.array(y_true)

x_train_data = []
for comment in x_train:
	x_train_data.append(model.infer_vector(comment))

x_test_data = []
for comment in x_test:
	x_test_data.append(model.infer_vector(comment))

classification_model = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
classification_model.fit(x_train_data, y_train)
#y_score = classification_model.decision_function(x_test_data)
#average_precision = average_precision_score(y_true_nd, y_score)
print "Naive Bayes : Accuracy on training data {}%".format(classification_model.score(x_train_data,y_train)*100)
print "Naive Bayes : Accuracy on testing data {}%".format(classification_model.score(x_test_data, y_test)*100)
#print('Average precision-recall score: {0:0.2f}'.format(average_precision))

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(x_train_data, y_train)
#y_score = clf.decision_function(x_test_data)
#average_precision = average_precision_score(y_true_nd, y_score)
print "Random Forest : Accuracy on training data {}%".format(clf.score(x_train_data,y_train)*100)
print "Random Forest : Accuracy on testing data {}%".format(clf.score(x_test_data, y_test)*100)
#print('Average precision-recall score: {0:0.2f}'.format(average_precision))

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(x_train_data, y_train)
#y_score = clf.decision_function(x_test_data)
#average_precision = average_precision_score(y_true_nd, y_score)
print "Neural Network : Accuracy on training data {}%".format(clf.score(x_train_data,y_train)*100)
print "Neural Network : Accuracy on testing data {}%".format(clf.score(x_test_data, y_test)*100)
#print('Average precision-recall score: {0:0.2f}'.format(average_precision))

from sklearn import svm
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(x_train_data, y_train)
y_score = clf.decision_function(x_test_data)
average_precision = average_precision_score(y_true_nd, y_score)

print "SVM : Accuracy on training data {}%".format(clf.score(x_train_data,y_train)*100)
print "SVM : Accuracy on testing data {}%".format(clf.score(x_test_data, y_test)*100)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))
#print "average precision" , average_precision
