from google.cloud import language_v1
from flask import Flask, jsonify

app = Flask(__name__)

# Instantiate the client
client = language_v1.LanguageServiceClient()

# Define the text for analysis
text = "I am feeling happy and excited about this vacation."

# Perform sentiment analysis
document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment

# Perform entity analysis
entities = client.analyze_entities(request={'document': document}).entities

# Print the sentiment score and magnitude
print('Sentiment Score:', sentiment.score)
print('Sentiment Magnitude:', sentiment.magnitude)

# Print the entities and their types
for entity in entities:
    print('Entity:', entity.name)
    print('Entity Type:', language_v1.Entity.Type(entity.type_).name)
    print('')

if __name__ == '__main__':
    # Run the app server on localhost:5000
    app.run('localhost', 5000)