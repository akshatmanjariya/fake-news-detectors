# app/preprocess.py
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Download if needed
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned = [
        lemmatizer.lemmatize(w)
        for w in words if w not in stop_words and len(w) > 2
    ]
    return ' '.join(cleaned)

def batch_clean_texts(texts):
    return [clean_text(t) for t in texts]

def get_vectorizer(X_train_clean):
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf.fit(X_train_clean)
    return tfidf

def vectorize_text(vectorizer, texts):
    return vectorizer.transform(texts)
