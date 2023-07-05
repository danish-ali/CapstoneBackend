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


# NewsChannel API key
api_key = 'e02243bc390540a3933687818730b8d0'

DB_HOST = '127.0.0.1'
DB_USER = 'root'
DB_PASSWORD = 'root'
DB_DATABASE = 'capstone'

# MySQL database configuration
#db_config = {
 #   'host': '127.0.0.1',
  #  'user': 'root',
   # 'password': 'root',
    #'database': 'capstone'
#}

# Connect to MySQL database
#db_conn = pymysql.connector.connect(**db_config)
#cursor = db_conn.cursor()

db_connection = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_DATABASE
)

cursor = db_connection.cursor()



# Set up NLTK resources
stop_words = set(stopwords.words("english"))
# we import the SentimentIntensityAnalyzer class from nltk.sentiment.vader to perform sentiment analysis using VADER.
#  The preprocess_text() function remains the same for data preprocessing.
sia = SentimentIntensityAnalyzer()
stemmer = PorterStemmer()

# The preprocess_text function is defined to preprocess the input text. It performs several text cleaning operations such as 
# converting the text to lowercase, removing URLs, eliminating non-alphanumeric characters, removing extra whitespaces, 
# tokenizing the text, removing stop words, and performing stemming. The preprocessed text is then returned.

def preprocess_text(text):
   # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove non-alphanumeric characters and extra whitespaces
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words and perform stemming
    tokens = [token for token in tokens if token not in stop_words]

    # Join tokens back into a preprocessed string
    preprocessed_text = " ".join(tokens)

# I have added a preprocess_text() function that applies various preprocessing steps to the text data. 
# The function converts the text to lowercase, removes URLs, non-alphanumeric characters, and extra whitespaces. 
# It then tokenizes the text, removes stop words, and performs lemmatization using NLTK resources.
#Within the /news endpoint, the article titles are preprocessed using the preprocess_text() 
#function before being returned as the JSON response.

#compound: The compound score represents the overall sentiment polarity of the text. It ranges from -1 (extremely negative) to 1 (extremely positive). In the output you shared, the compound score is -0.5267, indicating a negative sentiment.
#neg: The negative score represents the negative sentiment intensity of the text. It also ranges from 0 to 1. In the output you shared, the negative score is 0.221, indicating a moderate amount of negative sentiment.
#neu: The neutral score represents the neutral sentiment intensity of the text. It ranges from 0 to 1 as well. In the output you shared, the neutral score is 0.779, indicating a relatively high neutral sentiment.
#pos: The positive score represents the positive sentiment intensity of the text. It ranges from 0 to 1. In the output you shared, the positive score is 0.0, indicating no positive sentiment.

    return preprocessed_text if preprocessed_text else " "



def newsEmotionsSingleGraphDBSave(country="us"):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=2)    
    sentiment_scores = {}
    print("batch job executed")

    # Fetch top headlines
    for single_date in (start_date + datetime.timedelta(n) for n in range((end_date - start_date).days + 1)):
        url = f"https://newsapi.org/v2/top-headlines?country={country}&apiKey={api_key}&from={single_date}&to={single_date}"
        response = requests.get(url)

        if response.status_code == 200:
            articles = response.json()["articles"]

            for article in articles:
                # Perform sentiment analysis
                title = article["title"]
                content = article["content"]
                preprocessed_title = preprocess_text(title)
                sentiment_score = sia.polarity_scores(preprocessed_title)

                news_source = article["source"]["name"]
                if news_source not in sentiment_scores:
                    sentiment_scores[news_source] = {
                        "compound": [],
                        "neg": [],
                        "neu": [],
                        "pos": []
                    }

                for emotion in sentiment_score:
                    sentiment_scores[news_source][emotion].append(sentiment_score.get(emotion, 0))

                # Store the information in the database
                insert_query = """
                    INSERT INTO news_emotions (source, title, content, date, compound, neg, neu, pos)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = (news_source, title, content, single_date, sentiment_score["compound"],
                          sentiment_score["neg"], sentiment_score["neu"], sentiment_score["pos"])
                cursor.execute(insert_query, values)
                db_connection.commit()

        else:
            print(f"Error fetching news for {single_date}: {response.text}")

    # Generate bar charts using stored data
    # ...

    return sentiment_scores

# Schedule the job to run every 60 minutes
schedule.every(60).minutes.do(newsEmotionsSingleGraphDBSave)

while True:
    schedule.run_pending()
    time.sleep(1)