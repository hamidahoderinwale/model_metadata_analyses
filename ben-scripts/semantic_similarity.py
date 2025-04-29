# Semantic Similarity Analysis with TF-IDF
# Copy this code into your Jupyter notebook

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Function to preprocess text
def preprocess_text(text):
    if pd.isna(text):
        return ""
    # Convert to string if it's not
    text = str(text)
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Remove single characters
    tokens = [token for token in tokens if len(token) > 1]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Preprocess model cards
df['processed_card'] = df['card'].apply(preprocess_text)

# Create TF-IDF vectors with custom parameters
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,  # Limit to top 1000 terms
    min_df=2,          # Ignore terms that appear in less than 2 documents
    max_df=0.95,       # Ignore terms that appear in more than 95% of documents
    stop_words='english',
    ngram_range=(1, 2) # Consider both single words and word pairs
)

# Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_card'])

# Get feature names (terms)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Calculate cosine similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# Create a DataFrame for the similarity matrix
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=df['model_id'],
    columns=df['model_id']
)

# Visualize similarity matrix
plt.figure(figsize=(12, 10))
sns.heatmap(
    similarity_df,
    annot=True,
    fmt='.2f',
    cmap='YlOrRd',
    square=True
)
plt.title('Semantic Similarity Between Models (TF-IDF)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Find most similar model pairs
similarity_values = []
for i in range(len(similarity_df)):
    for j in range(i+1, len(similarity_df)):
        model1 = similarity_df.index[i]
        model2 = similarity_df.columns[j]
        similarity = similarity_df.iloc[i, j]
        similarity_values.append({
            'model1': model1,
            'model2': model2,
            'similarity': similarity
        })

# Create DataFrame of similarities
similarity_pairs = pd.DataFrame(similarity_values)
similarity_pairs = similarity_pairs.sort_values('similarity', ascending=False)

# Display top 5 most similar pairs
print("Top 5 Most Similar Model Pairs:")
similarity_pairs.head()

# Analyze similarity by depth difference
similarity_pairs['depth_diff'] = similarity_pairs.apply(
    lambda row: abs(
        df[df['model_id'] == row['model1']]['depth'].iloc[0] - 
        df[df['model_id'] == row['model2']]['depth'].iloc[0]
    ),
    axis=1
)

# Plot similarity vs depth difference
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=similarity_pairs,
    x='depth_diff',
    y='similarity',
    alpha=0.6
)
plt.title('Semantic Similarity vs Depth Difference (TF-IDF)')
plt.xlabel('Depth Difference')
plt.ylabel('Semantic Similarity')
plt.grid(True)
plt.show()

# Display top terms for each model
def get_top_terms(tfidf_matrix, feature_names, model_index, top_n=10):
    row = tfidf_matrix[model_index].toarray()[0]
    top_indices = row.argsort()[-top_n:][::-1]
    return [(feature_names[i], row[i]) for i in top_indices]

print("\nTop terms for each model:")
for i, model_id in enumerate(df['model_id']):
    top_terms = get_top_terms(tfidf_matrix, feature_names, i)
    print(f"\n{model_id}:")
    for term, score in top_terms:
        print(f"  {term}: {score:.4f}") 