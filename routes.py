from flask import Blueprint, render_template, request
from .model import predict

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        news_text = request.form['news_input']
        label, conf = predict(news_text)
        return render_template(
            'result.html',
            prediction=label, 
            confidence=conf, 
            text=news_text
        )
    return render_template('index.html')
