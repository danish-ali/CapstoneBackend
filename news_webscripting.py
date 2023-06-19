import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify

app = Flask(__name__)

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

if __name__ == '__main__':
 #   app.run(debug=True)
    app.run('localhost', 5000)