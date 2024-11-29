# 🎥 Graph-Based Movie Recommendation System

![System Architecture](Graphdbmr.drawio.svg)

## 📌 Introduction

The **Graph-Based Movie Recommendation System** is an intelligent application designed to provide personalized movie recommendations. Leveraging the power of **Neo4j Graph Database** and **Machine Learning** techniques, this system delivers highly accurate suggestions by analyzing multiple facets of movies, including plot, genres, actors, directors, and keywords. The application is deployed on Streamlit Cloud, offering a seamless and interactive user experience.

## 🌟 Key Features

- **Graph-Driven Data Modeling**: 
   - Movies, genres, actors, directors, and keywords are represented as nodes in a graph database, with relationships depicting real-world connections.
- **Multi-Dimensional Recommendations**: 
   - Incorporates various factors such as plot similarity, shared genres, actor collaborations, director influence, and thematic keywords.
- **Customizable Similarity Weights**: 
   - Allows users to adjust the importance of each factor dynamically, tailoring recommendations to their preferences.
- **Scalable and Interactive**: 
   - Deployed on Streamlit Cloud, making it easily accessible and responsive.

## 📐 System Architecture

The architecture is built around a **graph database design**, with Neo4j serving as the backbone for storing and querying relational data. A **Streamlit** interface interacts with the Neo4j database and processes data using **machine learning algorithms** like TF-IDF vectorization and cosine similarity to generate accurate recommendations.

Key Components:
- **Graph Database**: Neo4j aura for modeling entities and relationships.
- **Similarity Computation**: TF-IDF for text analysis, Jaccard similarity for categorical data, and cosine similarity for vector-based comparisons.
- **Streamlit UI**: User-friendly interface for exploring recommendations.

## 🌐 How to Use

The application is deployed live and can be accessed directly: [Graph-Based Movie Recommendation System](https://graph4recommendations.streamlit.app/).

### Steps to Use:
1. **Enter a Movie Title**: Input the name of a movie you like.
2. **Adjust Similarity Weights**: Fine-tune weights for plot, genre, actor, director, and keyword similarity using sliders.
3. **Generate Recommendations**: View a ranked list of movies similar to your chosen title.

## 🛠 Technical Stack

- **Frontend**: Streamlit for an interactive and responsive user interface.
- **Backend**: Neo4j Graph Database for storing and querying interconnected data.
- **Data Analysis and Processing**: 
  - **Python Libraries**: NLTK, Scikit-learn, and Pandas.
  - **Machine Learning**: TF-IDF vectorization, cosine similarity, and Jaccard similarity.

## 🔧 Deployment

The application is hosted on **Streamlit Cloud** and is accessible via this link: [https://graph4recommendations.streamlit.app/](https://graph4recommendations.streamlit.app/). The deployment ensures high availability and real-time responsiveness.

## 🏆 Acknowledgments

This project was made possible with:
- **Neo4j**: A powerful platform for graph-based data modeling.
- **Streamlit**: Enabling rapid development of interactive applications.
- **Python Libraries**: Essential tools for data processing and analysis, including NLTK, Scikit-learn, and Pandas.

---

Elevate your movie-watching experience with this state-of-the-art recommendation system! 🎬

