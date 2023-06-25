import json
from flask import Flask, jsonify
import requests
from textblob import TextBlob
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from flask import request
import tweepy
import re
import nltk
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# apply ML naive bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
from collections import defaultdict
import numpy as np
import mysql.connector

app = Flask(__name__)
# app.run(debug=True)
CORS(app)


consumer_key = 'Dq1VKypKj2s8uBBdLJsp5UVZM'
consumer_secret = 'I0eV5YO6GApFZNK1y5t46NutKw5VNI7UfgKJwFrJC9PUgleK0p'
access_token = '1625128811405754375-M6ieUWwRQYVYcQyRlJHC63VEJsP3GA'
access_token_secret = 'LfBwwtQSt77kC2IyOkiiYSUpsSSP83iksaZxjoCBRCeyI'

# NewsChannel API key
api_key = 'e02243bc390540a3933687818730b8d0'

# Authenticate with Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
twitter_api = tweepy.API(auth)


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

@app.route("/news", methods=["GET"])
def get_news():
    country = request.args.get("country", default="us")
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=10)
    sentiment_scores = []
    titles = []
    dates = []

    # Fetch top headlines
    for single_date in (start_date + datetime.timedelta(n) for n in range((end_date - start_date).days + 1)):
        url = f"https://newsapi.org/v2/top-headlines?country={country}&apiKey={api_key}&from={single_date}&to={single_date}"
        response = requests.get(url)

        if response.status_code == 200:
            articles = response.json()["articles"]

            for article in articles:
                # Perform sentiment analysis and append sentiment scores to the list
                title = article["title"]
                preprocessed_title = preprocess_text(title)
                sentiment_score = sia.polarity_scores(preprocessed_title)['compound']
                sentiment_scores.append(sentiment_score)
                titles.append(title)
                dates.append(single_date.strftime("%Y-%m-%d"))

        else:
            print(f"Error fetching news for {single_date}: {response.text}")

    # Extract relevant features from preprocessed data
    # Convert sentiment scores to binary labels
    y_train_binary = [1 if sentiment >= 0 else 0 for sentiment in sentiment_scores]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(titles, y_train_binary, test_size=0.2, random_state=42)

    # Create TF-IDF feature vectors
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Train the Naive Bayes classifier
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train_tfidf, y_train)

    # Make predictions on the test set
    y_pred = naive_bayes.predict(X_test_tfidf)

    # Evaluate the performance of the Naive Bayes classifier
    print(classification_report(y_test, y_pred))

    # Extract word frequencies
    count_vectorizer = CountVectorizer()
    X_word_frequencies = count_vectorizer.fit_transform(titles)
    word_frequencies = X_word_frequencies.toarray()

    # Extract n-grams
    ngram_vectorizer = CountVectorizer(ngram_range=(2, 3))
    X_ngrams = ngram_vectorizer.fit_transform(titles)
    ngrams = X_ngrams.toarray()

    # Extract TF-IDF values
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(titles)
    tfidf = X_tfidf.toarray()

     # Extract specific date field for x-axis values
    extracted_dates = list(set(dates))

    # Sort the dates in ascending order
    extracted_dates.sort()

    # Create a list of sentiment scores corresponding to each date
    averaged_sentiment_scores = [
        sum(score for date, score in zip(dates, sentiment_scores) if date == extracted_date) / dates.count(extracted_date)
        for extracted_date in extracted_dates
    ]

    # Generate line chart to visualize emotional levels over time
    # Positive sentiment scores indicate more positive or optimistic emotions, while negative scores indicate more negative
    # or pessimistic emotions.
    plt.plot(extracted_dates, averaged_sentiment_scores)
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.title("Emotional Levels Over Time")
    plt.xticks(rotation=45)
    plt.savefig("C:\\Users\\danis\\OneDrive\\Desktop\\MS\\CST-590\\graphs\\figure.png")

    return jsonify({
        "sentiment_scores": averaged_sentiment_scores,
        "word_frequencies": word_frequencies.tolist(),
        "ngrams": ngrams.tolist(),
        "tfidf": tfidf.tolist()
    })








@app.route("/newsEmotionsSingleGraph", methods=["GET"])
def newsEmotionsSingleGraph():
    country = request.args.get("country", default="us")
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=15)
    sentiment_scores = {}
    titles = []
    dates = []

    # Fetch top headlines
    for single_date in (start_date + datetime.timedelta(n) for n in range((end_date - start_date).days + 1)):
        url = f"https://newsapi.org/v2/top-headlines?country={country}&apiKey={api_key}&from={single_date}&to={single_date}"
        response = requests.get(url)

        if response.status_code == 200:
            articles = response.json()["articles"]

            for article in articles:
                # Perform sentiment analysis and append sentiment scores to the list
                title = article["title"]
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

                titles.append(title)
                dates.append(single_date.strftime("%Y-%m-%d"))

        else:
            print(f"Error fetching news for {single_date}: {response.text}")

    # Extract specific date field for x-axis values
    extracted_dates = list(set(dates))

    # Sort the dates in ascending order
    extracted_dates.sort()

    # Prepare data for the bar charts
    emotions = ["compound", "neg", "neu", "pos"]  # Specify the emotions
    num_emotions = len(emotions)  # Number of emotions
    num_sources = len(sentiment_scores)  # Number of news sources

    # Set the width and spacing for each bar
    bar_width = 0.2
    spacing = 0.05

    # Set the index for each news source
    index = range(num_sources)

    # Set the colors for each emotion
    colors = ['blue', 'green', 'orange', 'red']

    # Generate multiple bar charts for each emotion
    for emotion in emotions:
        scores = []
        for news_source in sentiment_scores:
            scores.append(sentiment_scores[news_source][emotion][0])

        # Create the bar chart
        plt.bar(index, scores, bar_width, label=emotion, color=colors)

    # Set the X-axis labels and tick positions
    plt.xlabel("News Source")
    plt.ylabel("Emotion Score")
    plt.title("Emotion Scores by News Source")
    plt.xticks(index, sentiment_scores.keys(), rotation=45)
    plt.legend()
    plt.tight_layout()

    # Save the bar chart as an image
    plt.savefig("C:\\Users\\danis\\OneDrive\\Desktop\\MS\\CST-590\\graphs\\newsEmotions.png")

    # Show the bar chart
    plt.show()

    return sentiment_scores




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

@app.route("/newsEmotionsSingleGraphDBSave", methods=["GET"])
def newsEmotionsSingleGraphDBSave():
    country = request.args.get("country", default="us")
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=2)
    sentiment_scores = {}

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



##### this is bind  with NewsEmotionsSingleGraph on UI
### getNewsEmotionsSingleGraphDB?start_date=2022-01-01&end_date=2022-12-31&news_source=The%20Tribune%20India

@app.route("/getNewsEmotionsSingleGraphDB", methods=["GET"])
def getNewsEmotionsSingleGraphDB():
    # Retrieve request parameters
    date = request.args.get("date")
    news_source = request.args.get("news_source")
    compound_value = request.args.get("compound_value")

    # Prepare the SQL query
    select_query = "SELECT * FROM news_emotions"
    where_clause = ""
    values = ()

    if date or news_source or compound_value:
        where_clause = " WHERE"
        conditions = []
        if date:
            conditions.append(" date = %s")
            values += (date,)
        if news_source:
            conditions.append(" source = %s")
            values += (news_source,)
        if compound_value:
            conditions.append(" compound = %s")
            values += (compound_value,)
        where_clause += " AND".join(conditions)

    full_query = select_query + where_clause
    cursor.execute(full_query, values)

    # Fetch the results
    results = cursor.fetchall()

    # Create a dictionary to store the news emotions
    news_emotions = {}

    # Process the results
    for row in results:
        source = row[1]
        if source not in news_emotions:
            news_emotions[source] = {
                'compound': [],
                'date': [],
                'neg': [],
                'neu': [],
                'pos': []
            }

        news_emotions[source]['compound'].append(float(row[5]))
        news_emotions[source]['date'].append(row[4].strftime("%Y-%m-%d"))
        news_emotions[source]['neg'].append(float(row[6]))
        news_emotions[source]['neu'].append(float(row[7]))
        news_emotions[source]['pos'].append(float(row[8]))

    # Return the news emotions as JSON
    return json.dumps(news_emotions)

 














@app.route("/newsEmotions", methods=["GET"])
def newsEmotions():
    country = request.args.get("country", default="us")    
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=2)
    sentiment_scores = {}
    titles = []
    dates = []

    # Fetch top headlines
    for single_date in (start_date + datetime.timedelta(n) for n in range((end_date - start_date).days + 1)):
        url = f"https://newsapi.org/v2/top-headlines?country={country}&apiKey={api_key}&from={single_date}&to={single_date}"
        response = requests.get(url)

        if response.status_code == 200:
            articles = response.json()["articles"]

            for article in articles:
                # Perform sentiment analysis and append sentiment scores to the list
                title = article["title"]
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

                titles.append(title)
                dates.append(single_date.strftime("%Y-%m-%d"))

        else:
            print(f"Error fetching news for {single_date}: {response.text}")

    # Extract specific date field for x-axis values
    extracted_dates = list(set(dates))

    # Sort the dates in ascending order
    extracted_dates.sort()

    # Prepare data for the bar charts
    emotions = list(sentiment_scores[next(iter(sentiment_scores))].keys())  # Get the list of emotions
    num_emotions = len(emotions)  # Number of emotions
    num_sources = len(sentiment_scores)  # Number of news sources

    # Set the width and spacing for each bar
    bar_width = 0.2
    spacing = 0.05

    # Set the index for each news source
    index = np.arange(num_sources)

    # Set the colors for each emotion
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan', 'magenta'][:num_emotions]

    # Generate separate bar charts for each emotion
    for i, emotion in enumerate(emotions):
        scores = [sentiment_scores[news_source][emotion][0] for news_source in sentiment_scores]
        plt.figure(figsize=(12, 6))  # Increase the figure size
        plt.bar(index, scores, bar_width, label=emotion, color=colors[i])
        plt.xlabel("News Source")
        plt.ylabel("Emotion Score")
        plt.title(f"{emotion.capitalize()} Scores by News Source")
        plt.xticks(index, sentiment_scores.keys(), rotation=45)
        plt.legend()
        plt.tight_layout()

        # Save the bar chart as an image
        plt.savefig(f"C:\\Users\\danis\\OneDrive\\Desktop\\MS\\CST-590\\graphs\\{emotion}_newsEmotions.png")

        # Show the bar chart
        plt.show()

    return sentiment_scores









@app.route("/tweets", methods=["GET"])
def get_tweets():
    keyword = request.args.get("keyword")
    count = request.args.get("count", default=10, type=int)

    # Fetch tweets
    tweets = twitter_api.search(q=keyword, count=count)

    # Extract tweet text
    tweet_texts = [tweet.text for tweet in tweets]

    return jsonify(tweet_texts)

@app.route('/newsService')
def get_newsService():
    topic = request.args.get("topic")
 # Fetch news articles for all topics
    url = f'https://api.newschannel.com/v1/articles?apiKey={api_key}'
    response = requests.get(url)

    # Parse response
    if response.status_code == 200:
        articles = response.json()["articles"]
        
        # Preprocess article titles and perform sentiment analysis
        sentiment_scores = []
        for article in articles:
            title = article["title"]
            preprocessed_title = preprocess_text(title)
            sentiment_score = sia.polarity_scores(preprocessed_title)
            sentiment_scores.append({
                "title": title,
                "message": sentiment_score["compound"],
                "source": article["source"]["name"]
            })

        return jsonify(sentiment_scores)
    else:
        return jsonify({"error": response.text}), response.status_code


#@app.route('/news')
#def get_news():
    # Make a request to the News API
#    url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}'
#    response = requests.get(url)

#    if response.status_code == 200:
#        news_data = response.json()
#        categorized_news = categorize_news(news_data)
#        return json.dumps(categorized_news)  # Serialize the data as JSON
#   else:
#        return jsonify({'error': 'Failed to fetch news data'})

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


@app.route('/get_bbc_news_comments')
def get_bbc_news_comments():
    # Fetch news articles from BBC RSS feed
    rss_url = 'http://feeds.bbci.co.uk/news/rss.xml'
    response = requests.get(rss_url)
    rss_data = response.text

    # Parse the RSS data
    soup = BeautifulSoup(rss_data, 'xml')
    articles = soup.find_all('item')

    news_with_comments = []

    # Iterate over the articles and extract user comments
    for article in articles:
        title = article.title.text
        link = article.link.text

        # Fetch the web page of the article
        response = requests.get(link)
        page_content = response.text

        # Parse the page content
        soup = BeautifulSoup(page_content, 'html.parser')

        # Extract user comments using appropriate CSS selectors or XPath expressions
        comments = soup.select('.comment-text')  # Example CSS selector, modify as per the website structure

        # Store the article title and extracted comments in a dictionary
        article_with_comments = {
            'title': title,
            'comments': [comment.get_text() for comment in comments]
        }

        news_with_comments.append(article_with_comments)

    return jsonify(news_with_comments)

@app.route('/get_washingtonpost_comments')
def get_washingtonpost_comments():
    # Fetch news articles from The Washington Post RSS feed
    rss_url = 'https://feeds.washingtonpost.com/rss/politics'
    response = requests.get(rss_url)
    rss_data = response.text

    # Parse the RSS data
    soup = BeautifulSoup(rss_data, 'xml')
    articles = soup.find_all('item')

    news_with_comments = []

    # Iterate over the articles and extract user comments
    for article in articles:
        title = article.title.text
        link = article.link.text

        # Fetch the web page of the article
        response = requests.get(link)
        page_content = response.text

        # Parse the page content
        soup = BeautifulSoup(page_content, 'html.parser')

        # Extract user comments using appropriate CSS selectors or XPath expressions
        comments = soup.select('.comments')  # Example CSS selector, modify as per the website structure

        for comment in comments:
            print(comment)  # Print the comment body
         
        # Select the parent element that contains the child div elements
        parent_element = soup.find('div', id='comments')

        if parent_element is not None:
            # Get all the child div elements
            child_divs = parent_element.find_all('div')

            # Extract the text from each child div
            text_list = [div.text.strip() for div in child_divs]

            # Print the extracted text
            for text in text_list:
                print(text)
        else:
            print("Parent element not found.")
        # Store the article title and extracted comments in a dictionary
        article_with_comments = {
            'title': title,
            'comments': [comment.get_text() for comment in comments]
        }

        news_with_comments.append(article_with_comments)

    return jsonify(news_with_comments)

@app.route('/cnn-instagram-posts')
def get_cnn_instagram_posts():
    # Set up ChromeDriver options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode


    # Set up Selenium web driver
    driver = webdriver.Chrome('C:\\Users\\danis\\Downloadschrome-win64', options=chrome_options)  # Replace with the path to your ChromeDriver

    try:
        # Navigate to CNN's Instagram profile
        driver.get('https://www.instagram.com/cnn/')

        # Extract post URLs using BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        post_urls = [a['href'] for a in soup.find_all('a', href=True) if '/p/' in a['href']]

        # Initialize a list to store the posts data
        posts_data = []

        # Iterate over post URLs
        for url in post_urls:
            # Navigate to the post page
            driver.get('https://www.instagram.com' + url)

            # Extract news content and comments using BeautifulSoup
            post_soup = BeautifulSoup(driver.page_source, 'html.parser')
            news_content = post_soup.find('div', class_='news-content').text.strip()
            comments = post_soup.find_all('div', class_='comment')
            
            # Extract the comment text from each comment element
            comment_text = [comment.text.strip() for comment in comments]

            # Create a dictionary with the post data
            post_data = {
                'news_content': news_content,
                'comments': comment_text
            }

            # Append the post data to the list
            posts_data.append(post_data)

        # Close the web driver
        driver.quit()

        # Return the posts data as a JSON response
        return jsonify(posts_data)

    except Exception as e:
        # Close the web driver in case of any exception
        driver.quit()
        return jsonify({'error': str(e)})




if __name__ == '__main__':
    # Run the app server on localhost:5000
    app.run('localhost', 5000)