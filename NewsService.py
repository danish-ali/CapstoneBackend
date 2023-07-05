import json
from flask import Flask, jsonify
import requests
from textblob import TextBlob
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
API_KEY = ''  # Replace with your News API key

@app.route('/news')
def get_news():
    # Make a request to the News API
    url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}'
    response = requests.get(url)

    if response.status_code == 200:
        news_data = response.json()
        categorized_news = categorize_news(news_data)
        return json.dumps(categorized_news)  # Serialize the data as JSON
    else:
        return jsonify({'error': 'Failed to fetch news data'})

def categorize_news(news_data):
    categorized_news = {
        'happy': 0,
        'sad': 0,
        'angry': 0,
        'optimistic': 0,
        'pessimistic': 0
    }

    # Extract articles from the news data
    articles = news_data.get('articles', [])

    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '')
        content = f'{title}. {description}'

        # Perform sentiment analysis using TextBlob
        blob = TextBlob(content)
        sentiment = blob.sentiment.polarity

        # Categorize article based on sentiment
        if sentiment > 0.2:
            categorized_news['happy'] += 1
        elif sentiment < -0.2:
            categorized_news['sad'] += 1
        elif sentiment > 0:
            categorized_news['optimistic'] += 1
        elif sentiment < 0:
            categorized_news['pessimistic'] += 1
        else:
            categorized_news['angry'] += 1

    return categorized_news

if __name__ == '__main__':
    # Run the app server on localhost:5000
    app.run('localhost', 5000)