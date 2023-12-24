import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import  classification_report

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

classification_rep = classification_report(y_test, y_pred)

print('Classification Report:')
print(classification_rep)

print("\nMeaning of Classification Report Columns:")
print("Precision: The ratio of true positive predictions to the total number of positive predictions.")
print("Recall: The ratio of true positive predictions to the total number of actual positive instances.")
print("F1-Score: The harmonic mean of precision and recall, providing a balance between the two.")
print("Support: The number of actual occurrences of the class in the specified dataset.")