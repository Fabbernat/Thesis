import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# Sample dataset (replace this with actual data)
data = {
    "text": [
        "This is a good movie",
        "I love this film",
        "It was an excellent performance",
        "I hated the storyline",
        "The plot was terrible",
        "I really enjoyed the characters",
        "It was a waste of time",
        "Brilliant acting and great script",
        "Horrible direction and bad dialogues",
        "One of the best movies I have seen"
    ],
    "label": [1, 1, 1, 0, 0, 1, 0, 1, 0, 1]  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# TF-IDF Vectorizer with unigrams and bigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')

# Classifier pipeline
model = make_pipeline(vectorizer, LogisticRegression())

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))
