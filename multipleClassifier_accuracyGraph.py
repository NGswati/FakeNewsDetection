import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns  



def preprocess_data(df, text_column, label_column):
    """
    Preprocess the data by removing empty data values.

    Parameters:
    - df: DataFrame, the input dataset.
    - text_column: str, the name of the column containing text data.
    - label_column: str, the name of the column containing labels.

    Returns:
    - preprocessed_df: DataFrame, the preprocessed dataset.
    """
    df = df.dropna(subset=[text_column, label_column]).reset_index(drop=True)

    label_encoder = LabelEncoder()
    df[label_column] = label_encoder.fit_transform(df[label_column])

    return df

df = pd.read_csv('news.csv')

text_column = 'text'
label_column = 'label'

preprocessed_df = preprocess_data(df, text_column, label_column)

X_train, X_test, y_train, y_test = train_test_split(preprocessed_df[text_column], preprocessed_df[label_column],test_size=0.2, random_state=42)

classifiers = [
    MultinomialNB(),
    SVC(),
    KNeighborsClassifier(),
    LogisticRegression(),
    RandomForestClassifier()
]

accuracies = []

for classifier in classifiers:
    model = make_pipeline(CountVectorizer(), classifier)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracies.append(accuracy)


plt.figure(figsize=(10, 6))
plt.plot([c.__class__.__name__ for c in classifiers], accuracies, marker='o', linestyle='-', color='b')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Different Classifiers')
plt.ylim(0, 1.0) 
plt.grid(True)
plt.show()