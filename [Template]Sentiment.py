import pandas as pd
from sentence_transformers import SentenceTransformer
import jieba
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from KeyMojiAPI import KeyMoji

# Load data and stopwords
comments_df = pd.read_csv('/Users/czu/Desktop/VSC/DataScience[NTU1131]/Project_BubbleTea/comments.csv')
stopwords_df = pd.read_csv('/Users/czu/Desktop/VSC/DataScience[NTU1131]/Project_BubbleTea/stopwords.csv', header=None)
basic_info_df = pd.read_csv('/Users/czu/Desktop/VSC/DataScience[NTU1131]/Project_BubbleTea/shop.csv')
stopwords = set(stopwords_df[0].tolist())

# Sentiment Analysis with KeyMoji
keymoji = KeyMoji(username="", keymojiKey="")

def analyze_sentiment(text):
    sense_result = keymoji.sense8(text, model="general", userDefinedDICT={"positive":[], "negative": [], "cursing": []})
    return {
        "品項": sense_result.get("品項", 0),
        "口味": sense_result.get("口味", 0),
        "服務態度": sense_result.get("服務態度", 0)
    }

for text in comments_df['text'].head(5):  # Adjust the number of samples as needed
    print(analyze_sentiment(text))

sentiment_scores = comments_df.groupby('shop_id')['text'].apply(lambda texts: {
    "品項": sum(analyze_sentiment(text)["品項"] for text in texts) / len(texts),
    "口味": sum(analyze_sentiment(text)["口味"] for text in texts) / len(texts),
    "服務態度": sum(analyze_sentiment(text)["服務態度"] for text in texts) / len(texts)
}).apply(pd.Series)
#sentiment_scores_melted = sentiment_scores.melt(id_vars=['shop_id'], var_name='category', value_name='score')
#sentiment_scores_melted.to_csv('???.csv')

# Scale sentiment scores
#scaler = MinMaxScaler(feature_range=(1, 5))
#sentiment_scores_scaled = pd.DataFrame(scaler.fit_transform(sentiment_scores), columns=sentiment_scores.columns, index=sentiment_scores.index)


sentiment_scores_matrix = sentiment_scores.reset_index()
sentiment_scores_matrix.columns = ['shop_id', 'categories', 'score']

# Display the first few rows of the matrix to ensure it matches the desired format
print(sentiment_scores_matrix.head())

df = pd.DataFrame(sentiment_scores_matrix)
df_pivot = df.pivot(index='shop_id', columns='categories', values='score').reset_index()
df_pivot.columns.name = None  # Remove the pivot name
print(df_pivot)
merge = pd.merge(basic_info_df, df_pivot, on = 'shop_id', how = 'left')
print(merge)


# Save output
#output_path = '/Users/czu/Desktop/VSC/DataScience[NTU1131]/Project_BubbleTea/trytry.csv'
#merge.to_csv(output_path, index=False)

print('All Done')
