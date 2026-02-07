# ===============================
# Fake News Detection - ML Project
# ===============================

import pandas as pd
import re
import nltk
import os
import pickle

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 1. Download stopwords (run once)
# -------------------------------
nltk.download('stopwords')

# -------------------------------
# 2. Load Dataset
# -------------------------------
fake = pd.read_csv("Dataset/Fake.csv")
real = pd.read_csv("Dataset/True.csv")

fake['label'] = 'FAKE'
real['label'] = 'REAL'

data = pd.concat([fake, real])
data = data.sample(frac=1).reset_index(drop=True)

# -------------------------------
# 3. Text Preprocessing Function
# -------------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

data['clean_text'] = data['text'].apply(clean_text)

# -------------------------------
# 4. Feature Extraction (TF-IDF)
# -------------------------------
X = data['clean_text']
y = data['label']

vectorizer = TfidfVectorizer(max_df=0.7)
X = vectorizer.fit_transform(X)

# -------------------------------
# 5. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 6. Model Training
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# 7. Prediction & Accuracy
# -------------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# 8. Save Model + Vectorizer
# -------------------------------
os.makedirs("model", exist_ok=True)
with open("model/fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
