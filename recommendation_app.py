import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# 1. Safe Dataset Loading
@st.cache_data
def load_data():
    movies = pd.read_csv("movies_for_app.csv")
    top_rated = pd.read_csv("top_rated_for_app.csv")
    
    # Handle missing 'overview' column
    if 'overview' not in movies.columns:
        movies['overview'] = ""
    movies['overview'] = movies['overview'].fillna('')
    
    # Handle missing 'soup' column dynamically
    if 'soup' not in movies.columns:
        meta_cols = [c for c in ['genres', 'keywords', 'cast', 'director', 'tagline'] if c in movies.columns]
        if meta_cols:
            movies['soup'] = movies[meta_cols].fillna('').astype(str).agg(' '.join, axis=1)
        else:
            movies['soup'] = movies['overview']
    else:
        movies['soup'] = movies['soup'].fillna('')
        
    return movies, top_rated

movies_df, top_rated_df = load_data()

# 2. Compute Matrices On The Fly
@st.cache_resource
def compute_matrices(df):
    # TF-IDF matrix for plot summaries
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    
    # Count matrix for metadata
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['soup'])
    
    # Similarity matrices
    sim_overview = cosine_similarity(tfidf_matrix, tfidf_matrix)
    sim_soup = cosine_similarity(count_matrix, count_matrix)
    
    return sim_overview, sim_soup

sim_overview, sim_soup = compute_matrices(movies_df)

indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

def get_recommendations(title, mode="Combined", top_n=10):
    if title not in indices:
        return pd.DataFrame()
    
    idx = indices[title]
    
    if mode == "Plot & Themes":
        sim_scores = list(enumerate(sim_overview[idx]))
    elif mode == "Cast & Genre Soup":
        sim_scores = list(enumerate(sim_soup[idx]))
    else:
        combined_sim = (sim_overview[idx] + sim_soup[idx]) / 2.0
        sim_scores = list(enumerate(combined_sim))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    display_cols = [c for c in ['title', 'vote_average', 'release_date', 'genres'] if c in movies_df.columns]
    return movies_df.iloc[movie_indices][display_cols]

# 3. UI Setup
st.title("🎬 Movie Recommendation System")

tab1, tab2 = st.tabs(["🎯 Get Recommendations", "🏆 Top Rated Movies"])

with tab1:
    selected_movie = st.selectbox("Select a movie you like:", movies_df['title'].values)
    mode = st.radio("Recommendation Strategy:", ["Combined", "Plot & Themes", "Cast & Genre Soup"], horizontal=True)
    
    if st.button("Recommend"):
        results = get_recommendations(selected_movie, mode=mode)
        if not results.empty:
            st.subheader(f"Movies similar to '{selected_movie}':")
            st.dataframe(results, use_container_width=True)
        else:
            st.error("Movie not found.")

with tab2:
    st.subheader("Top Rated Movies")
    display_top_cols = [c for c in ['title', 'vote_average', 'vote_count', 'genres'] if c in top_rated_df.columns]
    st.dataframe(top_rated_df[display_top_cols].head(20), use_container_width=True)
