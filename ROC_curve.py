import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def preprocess_data(df, text_column, label_column):
    
    df = df.dropna(subset=[text_column, label_column]).reset_index(drop=True)

    label_encoder = LabelEncoder()
    df[label_column] = label_encoder.fit_transform(df[label_column])

    return df

df = pd.read_csv('news.csv')

text_column = 'text'
label_column = 'label'

preprocessed_df = preprocess_data(df, text_column, label_column)

X_train, X_test, y_train, y_test = train_test_split(preprocessed_df[text_column], preprocessed_df[label_column], test_size=0.2, random_state=42)

model = make_pipeline(CountVectorizer(), LogisticRegression())
model.fit(X_train, y_train)


predictions = model.predict(X_test)

from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, predictions):.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

