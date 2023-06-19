import praw

# Create a Reddit API instance
reddit = praw.Reddit(
    client_id='YOUR_CLIENT_ID',
    client_secret='YOUR_CLIENT_SECRET',
    user_agent='YOUR_USER_AGENT',
)

# Define the news channel as a Reddit user
news_channel = 'BBCNews'  # Replace with the username of the news channel

# Get the Reddit user
redditor = reddit.redditor(news_channel)

# Collect the comments from the Reddit user's posts
comments = redditor.comments.new(limit=None)  # Replace `limit` with the desired number of comments

# Iterate over the comments
for comment in comments:
    print(comment.body)  # Print the comment body
    print('---')