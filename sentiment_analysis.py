import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('reviews.csv', quotechar='"')
# Make sure this file is in the same folder

# Function to analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return pd.Series([polarity, subjectivity, sentiment])

# Apply sentiment analysis
df[['Polarity', 'Subjectivity', 'Sentiment']] = df['review_text'].apply(analyze_sentiment)

# Display results
print(df[['review_text', 'Polarity', 'Subjectivity', 'Sentiment']])

# Visualize sentiment distribution
df['Sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.show()
