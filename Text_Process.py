import pandas as pd
from sentence_transformers import SentenceTransformer
import jieba


## Tokenization & Vectorization------------------------------------------------#
# Load data and stopwords
comments_df = pd.read_csv('/Users/czu/Desktop/VSC/DataScience[NTU1131]/Project_BubbleTea/shop_comments_with_id - Sheet1.csv')
stopwords_df = pd.read_csv('/Users/czu/Desktop/VSC/DataScience[NTU1131]/Project_BubbleTea/stopwords.csv', header=None)
basic_info_df = pd.read_csv('/Users/czu/Desktop/VSC/DataScience[NTU1131]/Project_BubbleTea/shop_basic_info - Sheet1.csv')
stopwords = set(stopwords_df[0].tolist())

# Initiate the model of SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Embeddings
comments = comments_df['text']  # Replace with actual column name
comments_df['embeddings'] = model.encode(comments, show_progress_bar=True).tolist()

def preprocess_comment(comment):
    tokens = jieba.lcut(comment)
    tokens = [word for word in tokens if word not in stopwords and len(word) > 1]  # Remove stopwords and single-character words
    return ' '.join(tokens)

comments_df['processed_comments'] = comments_df['text'].apply(preprocess_comment)

## Train the LDA model----------------------------------------------------------#
from sklearn.cluster import KMeans
import numpy as np

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(np.array(comments_df['embeddings'].tolist()))
comments_df['category'] = kmeans.labels_

## Display word list with highest frequency in each categories------------------#
from sklearn.metrics import pairwise_distances

top_n_comments = 200  # Number of closest comments to each cluster to analyze
top_n_words = 50     # Number of top words to extract

# Initialize a dictionary to store top words per category
top_words_per_category = {}

for i in range(n_clusters):
    # Find distances of comments to cluster center
    cluster_center = kmeans.cluster_centers_[i]
    distances = pairwise_distances([cluster_center], np.array(comments_df['embeddings'].tolist()), metric='cosine').flatten()
    
    # Get the indices of the closest comments
    closest_comment_indices = distances.argsort()[:top_n_comments]
    closest_comments = comments_df.iloc[closest_comment_indices]['processed_comments']
    
    # Tokenize and get top words
    words = []
    for comment in closest_comments:
        words.extend(comment.split())
    
    # Count word frequencies
    word_counts = pd.Series(words).value_counts()
    top_words_per_category[i] = word_counts.head(top_n_words).index.tolist()

# Display top words for each category
for category, words in top_words_per_category.items():
    print(f"Category {category} Top Words:", words)

top_words_data = []
for category, words in top_words_per_category.items():
    top_words_data.append({"Category": category, "Top Words": ', '.join(words)})
top_words_df = pd.DataFrame(top_words_data)
top_words_df.to_csv('/Users/czu/Desktop/VSC/DataScience[NTU1131]/Project_BubbleTea/top_words.csv', index=False)


## Define categories & label comments ---------------------------------------------#
from scipy.spatial.distance import cdist

# Define category names
category_names = {0: '品項', 1: '口味', 2: '服務態度'}

# Calculate distances to each cluster center
distances = cdist(np.array(comments_df['embeddings'].tolist()), kmeans.cluster_centers_, 'cosine')

# Get the closest two categories for each comment
closest_two_categories = distances.argsort(axis=1)[:, :2]

# Assign category names
comments_df['top_2_categories'] = closest_two_categories.tolist()
comments_df['top_2_category_names'] = comments_df['top_2_categories'].apply(lambda x: [category_names[x[0]], category_names[x[1]]])
comments_df.to_csv('/Users/czu/Desktop/VSC/DataScience[NTU1131]/Project_BubbleTea/labeled_comments.csv', index = False)

