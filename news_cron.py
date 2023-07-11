import schedule
import time
import requests
import json
from textblob import TextBlob
import mysql.connector
import datetime
import numpy as np
import re
import nltk
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer
from prometheus_client import start_http_server, Counter, Gauge
import configparser

# Read the API key from a property file
config = configparser.ConfigParser()
config.read('config.ini') 

# NewsChannel API key
api_key = config.get('API_KEYS', 'api_key')

DB_HOST = '127.0.0.1'
DB_USER = 'root'
DB_PASSWORD = 'root'
DB_DATABASE = 'capstone'

# MySQL database configuration
db_connection = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_DATABASE
)

cursor = db_connection.cursor()

# Prometheus metrics
total_articles_counter = Counter('news_articles_total', 'Total number of news articles processed')
success_articles_counter = Counter('news_articles_success', 'Number of successfully processed news articles')
error_articles_counter = Counter('news_articles_error', 'Number of news articles that encountered an error')
processing_time_gauge = Gauge('news_processing_time_seconds', 'Time taken to process news articles')

# Set up NLTK resources
stop_words = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    preprocessed_text = " ".join(tokens)
    return preprocessed_text if preprocessed_text else " "

def process_news_articles(country="us"):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=10)
    print("Batch job executed")

    for single_date in (start_date + datetime.timedelta(n) for n in range((end_date - start_date).days + 7)):
        url = f"https://newsapi.org/v2/top-headlines?country={country}&apiKey={api_key}&from={single_date}&to={single_date}"
        response = requests.get(url)

        if response.status_code == 200:
            articles = response.json()["articles"]

            for article in articles:
                total_articles_counter.inc()  # Increment the total articles counter

                try:
                    processing_start_time = time.time()  # Start measuring the processing time

                    title = article["title"]
                    content = article["content"]
                    preprocessed_title = preprocess_text(title)
                    sentiment_score = sia.polarity_scores(preprocessed_title)

                    news_source = article["source"]["name"]

                    # Store the information in the database
                    insert_query = """
                        INSERT INTO news_emotions (source, title, content, date, compound, neg, neu, pos)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    values = (news_source, title, content, single_date, sentiment_score["compound"],
                              sentiment_score["neg"], sentiment_score["neu"], sentiment_score["pos"])
                    cursor.execute(insert_query, values)
                    db_connection.commit()

                    success_articles_counter.inc()  # Increment the success articles counter

                    processing_time = time.time() - processing_start_time
                    processing_time_gauge.set(processing_time)  # Set the processing time gauge

                except Exception as e:
                    error_articles_counter.inc()  # Increment the error articles counter
                    print(f"Error processing news article: {e}")

        else:
            print(f"Error fetching news for {single_date}: {response.text}")

def run_scheduler():
    # Start the Prometheus HTTP server on port 8000
    start_http_server(8000)

    # Schedule the job to run every 60 minutes
    schedule.every(1).minutes.do(process_news_articles)

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    run_scheduler()
