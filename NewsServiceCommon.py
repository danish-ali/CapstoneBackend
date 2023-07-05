from flask import Flask, jsonify, request
import tweepy
import requests

app = Flask(__name__)
CORS(app)
# Twitter API credentials

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

# NewsChannel API key
api_key = ''

# Authenticate with Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
twitter_api = tweepy.API(auth)

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
def get_news():
    topic = request.args.get("topic")

    # Fetch news articles
    url = f"https://api.newschannel.com/v1/articles?apiKey={api_key}"
    response = requests.get(url)

    # Parse response
    if response.status_code == 200:
        articles = response.json()["articles"]
        article_titles = [article["title"] for article in articles]
        return jsonify(article_titles)
    else:
        return jsonify({"error": response.text}), response.status_code

if __name__ == "__main__":
     app.run('localhost', 5000)
