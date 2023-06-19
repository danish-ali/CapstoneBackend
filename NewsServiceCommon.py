from flask import Flask, jsonify, request
import tweepy
import requests

app = Flask(__name__)
CORS(app)
# Twitter API credentials

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
