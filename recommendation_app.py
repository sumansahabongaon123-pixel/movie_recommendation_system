import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# 1. Load Datasets
@st.cache_data
def load_data():
    movies = pd.read_csv("movies_for_app.csv")
    top_rated = pd.read_csv("top_rated_for_app.csv")
    return movies, top_rated

movies_df, top_rated_df = load_data()

# 2. Compute Matrices On The Fly
@st.cache_resource
def compute_matrices(df):
    # TF-IDF matrix for plot summaries
    tfidf = TfidfVectorizer(stop_words='english')
    df['overview'] = df['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    
    # Count matrix for metadata soup (cast, director, genres, keywords)
    count = CountVectorizer(stop_words='english')
    df['soup'] = df['soup'].fillna('')
    count_matrix = count.fit_transform(df['soup'])
    
    # Cosine similarity matrices
    sim_overview = cosine_similarity(tfidf_matrix, tfidf_matrix)
    sim_soup = cosine_similarity(count_matrix, count_matrix)
    
    return sim_overview, sim_soup

sim_overview, sim_soup = compute_matrices(movies_df)

# Map movie titles to indices
indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

def get_recommendations(title, mode="Combined", top_n=10):
    if title not in indices:
        return pd.DataFrame()
    
    idx = indices[title]
    
    if mode == "Plot & Themes":
        sim_scores = list(enumerate(sim_overview[idx]))
    elif mode == "Cast & Genre Soup":
        sim_scores = list(enumerate(sim_soup[idx]))
    else:  # Combined
        combined_sim = (sim_overview[idx] + sim_soup[idx]) / 2.0
        sim_scores = list(enumerate(combined_sim))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    return movies_df.iloc[movie_indices][['title', 'vote_average', 'release_date']]

# 3. Streamlit UI
st.title("🎬 Movie Recommendation System")

tab1, tab2 = st.tabs(["🎯 Get Recommendations", "🏆 Top Rated Movies"])

with tab1:
    selected_movie = st.selectbox("Select a movie you like:", movies_df['title'].values)
    mode = st.radio("Recommendation Strategy:", ["Combined", "Plot & Themes", "Cast & Genre Soup"], horizontal=True)
    
    if st.button("Recommend"):
        results = get_recommendations(selected_movie, mode=mode)
        if not results.empty:
            st.subheader(f"Movies similar to '{selected_movie}':")
            st.dataframe(results.rename(columns={'title': 'Title', 'vote_average': 'Rating', 'release_date': 'Release Date'}), use_container_width=True)
        else:
            st.error("Movie not found.")

with tab2:
    st.subheader("Top Rated Movies (Weighted Score)")
    st.dataframe(top_rated_df[['title', 'vote_average', 'vote_count']].head(20), use_container_width=True)
