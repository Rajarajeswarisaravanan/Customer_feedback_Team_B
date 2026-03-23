import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    
    if polarity > 0:
        return "positive", polarity
    elif polarity < 0:
        return "negative", polarity
    else:
        return "neutral", polarity


if __name__ == "__main__":
    # Read the dataset
    df = pd.read_csv("Milestone1_cleaned_feedback.csv")

    # Apply sentiment analysis
    df[["sentiment", "confidence_score"]] = df["clean_feedback"].apply(
        lambda x: pd.Series(get_sentiment(x))
    )

    # Save results
    df.to_csv("Milestone2_Sentiment_Results_new.csv", index=False)

    print("Milestone 2 completed successfully!")

    sentiment_counts = df['sentiment'].value_counts()

    plt.figure(figsize=(8,5))
    colors= ['green','red','gray']
    sentiment_counts.plot(kind='bar',color=colors)

    plt.title('How cutomer Feel - Sentiment Summary', fontsize=14)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.xticks(rotation=0)

    
    for i, count in enumerate(sentiment_counts):
        plt.text(i, count + 90, str(count), ha='center', fontsize=11, color='black')

    plt.savefig('sentiment-bar-chart.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("Bar chart saved as:sentiment-bar-chart.png")
    


    print(df[["clean_feedback", "sentiment", "confidence_score"]].head())

