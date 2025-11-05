import joblib
from .preprocess import clean_text

MODEL_PATH = 'models/fakenews_model.pkl'
VEC_PATH = 'models/tfidf_vectorizer.pkl'

def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return model, vectorizer

def predict(text):
    model, vectorizer = load_model()
    clean = clean_text(text)
    x_vec = vectorizer.transform([clean])
    pred = model.predict(x_vec)[0]
    prob = model.predict_proba(x_vec)[0].max()
    return pred, prob
