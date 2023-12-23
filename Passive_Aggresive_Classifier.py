import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df=pd.read_csv('news.csv')
print(df.shape)
df.head()

labels=df.label
labels.head()

x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.1, random_state=7)

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

PAC=PassiveAggressiveClassifier(max_iter=50,validation_fraction=0.1)
PAC.fit(tfidf_train,y_train)
y_pred=PAC.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)

print('Accuracy:' ,{round(score*100,2)} )

confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])