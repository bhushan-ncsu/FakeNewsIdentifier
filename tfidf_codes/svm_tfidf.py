import pandas
import numpy as np
from sklearn.metrics import average_precision_score

data = pandas.read_csv('fake_or_real_news.csv')
data.drop('id',axis=1,inplace=True)
data.drop('label',axis=1,inplace=True)
#data = data['title'].append(data['text'])
#data = data.apply(lambda x: x.lower())

##CREATE X axis
x_data = []
for i in range (0, len(data), 1):
	combined = ''.join([data['title'][i] , " ", data['text'][i]])
	x_data.append(combined)
	#print i

data = x_data
data = map(str.lower,data)


##CREATE Y axis
df = pandas.read_csv('fake_or_real_news.csv')
df = df['label']


##CREATE stratified data
from sklearn.model_selection import train_test_split
train, test, df_train, df_test = train_test_split(data, df, stratify=df, test_size=0.2)

y_true = []
for i in (df_test):
	if(i != "REAL"):
		y_true.append(0)
	else:
		y_true.append(1)

y_true_nd = np.array(y_true)

##CREATE TFIDF VECTOR for training data and transfor testing data in that vector
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
train_vector = tfidf.fit_transform(train)
test_vector = tfidf.transform(test)


##APPLY Model
from sklearn import svm
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(train_vector,df_train)

y_score = clf.decision_function(test_vector)
average_precision = average_precision_score(y_true_nd, y_score)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))

#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(clf, data, df, cv=5)

#print scores

print "SVM accuracy on training data : {}%".format(clf.score(train_vector,df_train)*100)

print "SVM accuracy on testing data : {}%".format(clf.score(test_vector, df_test)*100)

