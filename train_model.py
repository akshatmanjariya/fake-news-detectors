
import sys
import os
sys.path.append(os.path.abspath('../app'))
import pandas as pd
from preprocess import clean_text, get_vectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load data
df_fake = pd.read_csv('../data/Fake.csv')
df_true = pd.read_csv('../data/True.csv')

df_fake['label'] = 0
df_true['label'] = 1

df = pd.concat([df_fake, df_true], ignore_index=True)
df['text'] = df['text'].astype(str)
X = df['text'].apply(clean_text)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)

vectorizer = get_vectorizer(X_train)
X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=500)
model.fit(X_train_vec, y_train)

os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/fakenews_model.pkl')
joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')
print(f'Model saved. Test accuracy: {model.score(X_test_vec, y_test):.3f}')
