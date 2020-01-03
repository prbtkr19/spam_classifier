#importing the datasets
import nltk
import pandas as pd
messages =pd.read_csv("F:\smsspamcollection\SMSSpamCollection",sep ='\t',names=["label","message"])




#cleaning text and preprocessing
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()
corpus =[]

for i in range(0,len(messages)):# most important step in clleaning
    review= re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

#creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
x=cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y = y.iloc[:,1].values

#train test results
from sklearn.model_selection import train_test_split
x_train ,x_test ,y_train, y_test = train_test_split(x,y,test_size=0.20 ,random_state=0)

#training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model= MultinomialNB().fit(x_train,y_train)

y_pred=spam_detect_model.predict(x_test)

#creating confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


#checking acccuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
