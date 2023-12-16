import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

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

X_train, X_test, y_train, y_test = train_test_split(preprocessed_df[text_column], preprocessed_df[label_column], test_size=0.2, random_state=42)

model = make_pipeline(CountVectorizer(), LogisticRegression())

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Logistic Regression Accuracy: {accuracy:.4f}')

cm = confusion_matrix(y_test, predictions)
print('Confusion Matrix:')
print(cm)

classification_rep = classification_report(y_test, predictions)
print('Classification Report:')
print(classification_rep)

print("\nMeaning of Classification Report Columns:")
print("Precision: The ratio of true positive predictions to the total number of positive predictions.")
print("Recall: The ratio of true positive predictions to the total number of actual positive instances.")
print("F1-Score: The harmonic mean of precision and recall, providing a balance between the two.")
print("Support: The number of actual occurrences of the class in the specified dataset.")