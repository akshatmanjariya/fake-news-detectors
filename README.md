# Fake News Detection Web Application

Production-ready Flask web app to classify input news as Fake or Real using NLP and ML.

## Setup

1. Prepare environment:
    - `python -m venv venv && source venv/bin/activate`
    - `pip install -r requirements.txt`

2. Train the model:
    - Run `python train_model.py`
      (creates models/fakenews_model.pkl and models/tfidf_vectorizer.pkl)

3. Start application:
    - `python run.py`
    - Open http://localhost:5000 in browser.

## Project folders
- `app/` - Main Flask app code and web templates
- `data/` - CSVs for training
- `models/` - Saved model and vectorizer

## Features
- Paste any news text or URL content for instant classification
- Returns "Real" or "Fake" with probability confidence
- Easily extend with Random Forest/XGBoost
