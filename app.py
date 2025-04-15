import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# Load and clean dataset
df = pd.read_excel("Movies Data.xlsx")

# Clean unwanted characters
df = df[~df['original_title'].astype(str).str.contains(r'[èåŠ‡åÅåÐ´ç‰ˆã€ŒéÃ]')]

# Fill and filter runtimes
df['runtime'] = df['runtime'].fillna(0).astype(int)
df = df[df['runtime'] != 0]

# Filter non-zero ratings
df = df[df['vote_average'] != 0]

# Features used
features = ["popularity", "vote_average", "runtime", "Drama & Emotion", "Entertainment", "Thriller & Mystery"]
X = df[features]

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Fit KNN model
knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
knn.fit(X_scaled)

# Function to get recommendation
def recommend_movie(min_rating=5, category="Entertainment", runtime=90):
    category_vector = [1 if category == cat else 0 for cat in ["Drama & Emotion", "Entertainment", "Thriller & Mystery"]]
    query_vector = pd.DataFrame([[100, min_rating, runtime] + category_vector], columns=features)
    query_scaled = scaler.transform(query_vector)
    distance, index = knn.kneighbors(query_scaled, n_neighbors=1)
    recommended_movie = df.iloc[index[0][0]][["original_title", "tagline", "overview", "runtime", "vote_average", "popularity"]]
    recommended_movie["runtime"] = f"{recommended_movie['runtime']} minutes"
    return recommended_movie

# Streamlit App
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")


# App Title
st.title("🎥 Movie Recommendation System")
st.markdown("Get personalized movie recommendations based on your mood and preferences!")

# User Inputs
category = st.selectbox("Select Movie Category", ["Drama & Emotion", "Entertainment", "Thriller & Mystery"])
min_rating = st.slider("Select Minimum IMDb Rating", 0, 10, 7)
runtime = st.slider("Select Maximum Runtime (in minutes)", 60, 240, 120)

if st.button("Recommend Movie"):
    movie = recommend_movie(min_rating=min_rating, category=category, runtime=runtime)
    st.success("Here's a movie you might enjoy:")
    st.markdown(f"### 🎬 {movie['original_title']}")
    st.markdown(f"- **Tagline**: {movie['tagline']}")
    st.markdown(f"- **Rating**: ⭐ {movie['vote_average']}")
    st.markdown(f"- **Runtime**: ⏱️ {movie['runtime']}")
    st.markdown(f"- **Popularity**: 🔥 {movie['popularity']}")
    st.markdown(f"**Overview:** {movie['overview']}")