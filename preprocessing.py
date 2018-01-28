#https://github.ncsu.edu/vsshinde/ReviewClassification/blob/master/train_comments_layer3.py <-- how to use this model.
import pandas as pd
from nltk.corpus import stopwords
import nltk
import re
import os
import random
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
nltk.download('stopwords')
fake_or_real_data = pd.read_csv('fake_or_real_news.csv')
y_label = fake_or_real_data['label'].tolist()
x_title = fake_or_real_data['title'].tolist()
x_text = fake_or_real_data['text'].tolist()
x_data = []
print stopwords.words('english')
for i in range (0, len(x_title), 1):
#for i in range(0, 111, 1):
	combined = ''.join([x_title[i] , " ", x_text[i]])
	combined = combined.lower()
	combined = re.sub('\W+',' ', combined )
	current_list = combined.split()
	#print len(current_list)
	processed_list = current_list
	print "Processing record " , i
	for word in current_list:
		if word in stopwords.words('english'):
			processed_list.remove(word)
	x_data.append(processed_list)
	#print len(processed_list)

#Split the data into training and test
from sklearn.model_selection import train_test_split
#x_features = data['X_Features']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_label, stratify=y_label, test_size=0.2)

import pickle

train_data_file = open("x_train", 'wb')
test_data_file = open("x_test", 'wb')
train_label_file = open("y_train", 'wb')
test_label_file = open("y_test", 'wb')

pickle.dump(x_train, train_data_file)
pickle.dump(x_test, test_data_file)
pickle.dump(y_train, train_label_file)
pickle.dump(y_test, test_label_file)

'''
i = 0
labeled_comments = []
for comment in x_data:
    #print comment
    sentence = LabeledSentence(words=comment, tags=["COMMENT_"+str(i)])
    labeled_comments.append(sentence)
    i += 1

#more dimensions mean more trainig them, but more generalized
num_features = 300
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

x_train = []
for i in range(5000):
    x_train.append(model.docvecs["COMMENT_"+str(i)])

x_test = []
for i in range(5001,len(x_title)):
    x_test.append(model.docvecs["COMMENT_"+str(i)])
    
y_train = y_label[0:5000]
y_test = y_label[5001:len(x_title)]

'''
