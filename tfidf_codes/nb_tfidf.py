import pandas
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
train, test, df_train, df_test = train_test_split(data, df, stratify=df, test_size=0.3)


##CREATE TFIDF VECTOR for training data and transfor testing data in that vector
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(analyzer='word', stop_words='english', max_features=22000)
train_vector = tfidf.fit_transform(train)
test_vector = tfidf.transform(test)


##APPLY Model
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
clf.fit(train_vector.todense(),df_train)

print "Naive Bayes accuracy on training data : {}%".format(clf.score(train_vector.todense(),df_train)*100)

print "Naive Bayes accuracy on testing data : {}%".format(clf.score(test_vector.todense(), df_test)*100)

