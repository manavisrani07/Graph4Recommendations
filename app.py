# app.py

import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Set up Neo4j connection using Streamlit secrets
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USER = st.secrets["NEO4J_USER"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Helper functions
def run_query(query, parameters=None):
    with driver.session() as session:
        result = session.run(query, parameters)
        df = pd.DataFrame([dict(record) for record in result])
    return df

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def fetch_data():
    # Fetch data from Neo4j
    plots_df = run_query("MATCH (m:Movie) RETURN m.imdb_id AS imdb_id, m.title AS title, m.plot AS plot")
    genres_df = run_query("MATCH (m:Movie)-[:BELONGS_TO_GENRE]->(g:Genre) RETURN m.imdb_id AS imdb_id, collect(g.name) AS genres")
    actors_df = run_query("MATCH (m:Movie)<-[:ACTED_IN]-(a:Actor) RETURN m.imdb_id AS imdb_id, collect(a.name) AS actors")
    directors_df = run_query("MATCH (m:Movie)<-[:DIRECTED]-(d:Director) RETURN m.imdb_id AS imdb_id, collect(d.name) AS directors")
    keywords_df = run_query("MATCH (m:Movie)-[:HAS_KEYWORD]->(k:Keyword) RETURN m.imdb_id AS imdb_id, collect(k.name) AS keywords")

    # Merge dataframes
    movies_df = plots_df.merge(genres_df, on='imdb_id', how='left')
    movies_df = movies_df.merge(actors_df, on='imdb_id', how='left')
    movies_df = movies_df.merge(directors_df, on='imdb_id', how='left')
    movies_df = movies_df.merge(keywords_df, on='imdb_id', how='left')

    # Fill NaN values
    movies_df['genres'] = movies_df['genres'].apply(lambda x: x if isinstance(x, list) else [])
    movies_df['actors'] = movies_df['actors'].apply(lambda x: x if isinstance(x, list) else [])
    movies_df['directors'] = movies_df['directors'].apply(lambda x: x if isinstance(x, list) else [])
    movies_df['keywords'] = movies_df['keywords'].apply(lambda x: x if isinstance(x, list) else [])
    movies_df['plot'] = movies_df['plot'].fillna('')

    # Preprocess plots
    movies_df['processed_plot'] = movies_df['plot'].apply(preprocess_text)

    return movies_df

def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    if not set1 or not set2:
        return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return float(len(intersection)) / len(union)

def get_recommendations(movie_title, movies_df, top_n=5):
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(movies_df['processed_plot'])

    # Find the index of the target movie
    target_idx = movies_df.index[movies_df['title'].str.lower() == movie_title.lower()]
    if len(target_idx) == 0:
        st.write(f"Movie '{movie_title}' not found in the database.")
        return pd.DataFrame()
    target_idx = target_idx[0]

    # Compute cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[target_idx:target_idx+1], tfidf_matrix).flatten()
    movies_df['plot_similarity'] = cosine_similarities

    # Compute additional similarities
    target_movie = movies_df.iloc[target_idx]
    movies_df['genre_similarity'] = movies_df['genres'].apply(lambda x: jaccard_similarity(target_movie['genres'], x))
    movies_df['actor_similarity'] = movies_df['actors'].apply(lambda x: jaccard_similarity(target_movie['actors'], x))
    movies_df['director_similarity'] = movies_df['directors'].apply(lambda x: jaccard_similarity(target_movie['directors'], x))
    movies_df['keyword_similarity'] = movies_df['keywords'].apply(lambda x: jaccard_similarity(target_movie['keywords'], x))

    # Compute overall similarity
    weights = {
        'plot_similarity': 0.4,
        'genre_similarity': 0.2,
        'actor_similarity': 0.2,
        'director_similarity': 0.1,
        'keyword_similarity': 0.1
    }
    movies_df['overall_similarity'] = (
        weights['plot_similarity'] * movies_df['plot_similarity'] +
        weights['genre_similarity'] * movies_df['genre_similarity'] +
        weights['actor_similarity'] * movies_df['actor_similarity'] +
        weights['director_similarity'] * movies_df['director_similarity'] +
        weights['keyword_similarity'] * movies_df['keyword_similarity']
    )

    # Exclude the target movie
    recommendations = movies_df[movies_df.index != target_idx]

    # Sort by overall similarity
    recommendations = recommendations.sort_values(by='overall_similarity', ascending=False)

    # Select top N recommendations
    top_recommendations = recommendations[['title', 'overall_similarity']].head(top_n)
    return top_recommendations

# Streamlit app
st.title("Movie Recommendation System")

movies_df = fetch_data()

movie_title = st.text_input("Enter a movie title to get recommendations:")

if st.button("Get Recommendations"):
    if movie_title:
        with st.spinner('Calculating recommendations...'):
            recommendations = get_recommendations(movie_title, movies_df, top_n=5)
            if not recommendations.empty:
                st.subheader(f"Top recommendations for '{movie_title}':")
                st.table(recommendations)
            else:
                st.write("No recommendations found.")
    else:
        st.write("Please enter a movie title.")

# Close the driver connection when the app is stopped
driver.close()
