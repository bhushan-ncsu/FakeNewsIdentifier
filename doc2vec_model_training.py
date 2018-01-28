import pandas as pd
from nltk.corpus import stopwords
import nltk
import re
import os
import random
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
import pickle

with open ('x_train', 'rb') as fp:
    x_train = pickle.load(fp)

#with open ('x_test', 'r') as fp:
#    x_test = pickle.load(fp)

#with open ('y_train', 'r') as fp:
#    y_train = pickle.load(fp)

#with open ('y_test', 'r') as fp:
#    y_test = pickle.load(fp)

i = 0
labeled_comments = []
for comment in x_train:
    #print comment
    sentence = LabeledSentence(words=comment, tags=["COMMENT_"+str(i)])
    labeled_comments.append(sentence)
    i += 1

#more dimensions mean more trainig them, but more generalized
num_features = 500
# Minimum word count threshold.
min_word_count = 1
# Number of threads to run in parallel.
#num_workers = multiprocessing.cpu_count()
num_workers = 2
# Context window length.
context_size = 10
# Downsample setting for frequent words.
#rate 0 and 1e-5 
#how often to use
downsampling = 1e-4

# Initialize model
model = Doc2Vec(min_count=min_word_count,
	window=context_size, 
	size=num_features,
	sample=downsampling,
	negative=5,
	workers=num_workers)

model.build_vocab(labeled_comments)

# Train the model
# This may take a bit to run #20 is better
for epoch in range(10):
    print "Training iteration %d" % (epoch)
    random.shuffle(labeled_comments)
    model.train(labeled_comments,total_examples=model.corpus_count, epochs=model.iter)
#save model
if not os.path.exists("trained"):
    os.makedirs("trained")
model.save(os.path.join("trained", "comments2vec.d2v"))
#load model
model = Doc2Vec.load(os.path.join("trained", "comments2vec.d2v"))

#x_train = []
#for i in range(5000):
#    x_train.append(model.docvecs["COMMENT_"+str(i)])

#x_test = []
#for i in range(5001,len(x_title)):
#    x_test.append(model.docvecs["COMMENT_"+str(i)])
    
#y_train = y_label[0:5000]
#y_test = y_label[5001:len(x_title)]



