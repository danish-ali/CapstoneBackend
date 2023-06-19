import schedule
import time
import requests
import json
from textblob import TextBlob
import mysql.connector

API_KEY = 'e02243bc390540a3933687818730b8d0'
DB_HOST = '127.0.0.1'
DB_USER = 'root'
DB_PASSWORD = 'root'
DB_DATABASE = 'capstone'

# Set up MySQL connection
db_connection = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_DATABASE
)

cursor = db_connection.cursor()

def save_news_to_db(news):

    
    # Prepare the SQL query
    sql = "INSERT INTO newscontent (sentiments, newsSourceID) VALUES (%s, %s)"

    # Iterate over the categorized news data and execute the SQL query
    for category, count in news.items():
        values = (category, newsSourceID)
        cursor.execute(sql, values)

    # Commit the changes to the database
    new_varnew_var = db_connection.commit()
    

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

def get_news():
    print("Fetching news...")
    # Make a request to the News API
    url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}'
    response = requests.get(url)

    if response.status_code == 200:
        print("News data fetched successfully")
        news_data = response.json()
        categorized_news = categorize_news(news_data)

        # Save the categorized news to the database
        save_news_to_db(categorized_news)
        print("News data saved to the database")
    else:
        print('Failed to fetch news data')

# Schedule the job to run every 2 minutes
schedule.every(2).minutes.do(get_news)

while True:
    schedule.run_pending()
    time.sleep(1)