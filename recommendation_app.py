"""
Simple Movie Recommender -- Streamlit app.

Expects a `model_artifacts/` folder next to this file (produced by running
`recommendation_sys_improved.ipynb`), containing:
    movies_for_app.csv, top_rated_for_app.csv,
    tfidf_overview_vectorizer.pkl,  tfidf_overview_matrix.pkl,
    count_soup_vectorizer.pkl,      count_soup_matrix.pkl,
    tfidf_combined_vectorizer.pkl,  tfidf_combined_matrix.pkl

No hardcoded machine-specific paths: everything is resolved relative to this
file's own location, so `streamlit run recommendation_app.py` works no matter
which directory you launch it from or whose machine it's on.
"""
import ast
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE_DIR, "model_artifacts")

MODEL_OPTIONS = {
    "Combined (recommended)": "combined",
    "Plot & themes only": "overview",
    "Cast, director & genres": "soup",
    "Combined + quality boost": "hybrid",
}
MODEL_DESCRIPTIONS = {
    "combined": "Blends the plot summary with genres, keywords, top cast, and director.",
    "overview": "Matches on plot/theme text only -- ignores who's in it or who made it.",
    "soup": "Matches on genres, keywords, cast, and director -- ignores the plot text.",
    "hybrid": "Same as Combined, then re-ranks the top candidates to favor better-rated movies.",
}


# --------------------------------------------------------------------- data
@st.cache_data
def load_movies():
    df = pd.read_csv(os.path.join(ART_DIR, "movies_for_app.csv"))
    df["genres_list"] = df["genres_list"].apply(safe_eval_list)
    df["cast_top3"] = df["cast_top3"].apply(safe_eval_list)
    return df


@st.cache_data
def load_top_rated():
    return pd.read_csv(os.path.join(ART_DIR, "top_rated_for_app.csv"))


@st.cache_resource
def load_matrices():
    return {
        "overview": joblib.load(os.path.join(ART_DIR, "tfidf_overview_matrix.pkl")),
        "soup": joblib.load(os.path.join(ART_DIR, "count_soup_matrix.pkl")),
        "combined": joblib.load(os.path.join(ART_DIR, "tfidf_combined_matrix.pkl")),
    }


def safe_eval_list(x):
    """movies_for_app.csv stores list columns as their Python repr string,
    e.g. "['Action', 'Adventure']" -- turn that back into a real list."""
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        v = ast.literal_eval(x)
        return v if isinstance(v, list) else []
    except (ValueError, SyntaxError):
        return []


# ------------------------------------------------------------ recommending
def recommend(movies, title_to_pos, title, matrix, top_n=10):
    idx = title_to_pos[title]
    sims = cosine_similarity(matrix[idx], matrix).ravel()
    order = np.argsort(-sims)
    order = order[order != idx][:top_n]
    return movies.loc[order].assign(similarity=sims[order])


def recommend_hybrid(movies, title_to_pos, title, matrix, top_n=10, pool=30, quality_weight=0.35):
    idx = title_to_pos[title]
    sims = cosine_similarity(matrix[idx], matrix).ravel()
    order = np.argsort(-sims)
    order = order[order != idx][:pool]

    cand = movies.loc[order].copy()
    cand["similarity"] = sims[order]
    sim_range = (cand["similarity"].max() - cand["similarity"].min()) or 1.0
    cand["sim_norm"] = (cand["similarity"] - cand["similarity"].min()) / sim_range
    q_range = (cand["weighted_rating"].max() - cand["weighted_rating"].min()) or 1.0
    cand["quality_norm"] = (cand["weighted_rating"] - cand["weighted_rating"].min()) / q_range
    cand["blend"] = (1 - quality_weight) * cand["sim_norm"] + quality_weight * cand["quality_norm"]
    return cand.sort_values("blend", ascending=False).head(top_n)


def get_recommendations(movies, title_to_pos, matrices, title, mode, top_n=10):
    if mode == "hybrid":
        return recommend_hybrid(movies, title_to_pos, title, matrices["combined"], top_n=top_n)
    return recommend(movies, title_to_pos, title, matrices[mode], top_n=top_n)


# ------------------------------------------------------------------- UI
def render_result_row(rank, row, query_director):
    cols = st.columns([0.5, 5, 1.5, 1])
    cols[0].markdown(f"**{rank}.**")
    with cols[1]:
        st.markdown(f"**{row['display_title']}**")
        genres = ", ".join(row["genres_list"][:3]) if row["genres_list"] else "—"
        st.caption(genres)
        if query_director and isinstance(row.get("director"), str) and row["director"] == query_director:
            st.caption(f"🎬 Also directed by {query_director}")
        with st.expander("Plot"):
            st.write(row["overview"] if isinstance(row["overview"], str) and row["overview"] else "No overview available.")
    year = int(row["release_year"]) if pd.notna(row["release_year"]) else "—"
    cols[2].markdown(f"{year}")
    cols[3].markdown(f"⭐ {row['vote_average']:.1f}")


def main():
    st.set_page_config(page_title="Simple Movie Recommender", layout="centered")
    st.title("🎬 Simple Movie Recommender")
    st.write("Find movies similar to one you already like.")

    if not os.path.isdir(ART_DIR):
        st.error(
            f"Couldn't find `model_artifacts/` next to this script (looked in `{ART_DIR}`). "
            "Run `recommendation_sys_improved.ipynb` first -- its last section exports exactly "
            "the files this app needs into that folder."
        )
        return

    movies = load_movies()
    top_rated = load_top_rated()
    matrices = load_matrices()
    title_to_pos = pd.Series(movies.index, index=movies["display_title"])
    movie_list = movies["display_title"].sort_values().tolist()

    tab_search, tab_top = st.tabs(["🔎 Find similar movies", "🏆 Top rated"])

    with tab_search:
        selected_movie = st.selectbox("Select a movie:", movie_list)
        model_label = st.radio("Recommend using:", list(MODEL_OPTIONS.keys()), index=0)
        mode = MODEL_OPTIONS[model_label]
        st.caption(MODEL_DESCRIPTIONS[mode])

        if st.button("Get recommendations", type="primary"):
            query_row = movies.loc[title_to_pos[selected_movie]]
            recs = get_recommendations(movies, title_to_pos, matrices, selected_movie, mode, top_n=10)

            st.subheader(f"Movies similar to {selected_movie}")
            query_director = query_row["director"] if isinstance(query_row["director"], str) else None
            for rank, (_, row) in enumerate(recs.iterrows(), start=1):
                render_result_row(rank, row, query_director)
                st.divider()

    with tab_top:
        st.subheader("Top 50 by weighted rating")
        st.caption("IMDB-style weighted rating -- balances a movie's own average against the "
                   "catalog average, so a handful of 10/10 votes can't outrank a genuinely "
                   "well-established favorite.")
        st.dataframe(
            top_rated[["display_title", "vote_average", "vote_count", "weighted_rating"]]
            .rename(columns={"display_title": "Title", "vote_average": "Rating",
                              "vote_count": "# Votes", "weighted_rating": "Weighted Rating"})
            .reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.caption(
        "Content-based filtering over the TMDB 5000 dataset. \"Combined\" and \"quality boost\" "
        "use plot text plus genres/keywords/cast/director; see the companion notebook for how "
        "each model compares."
    )


if __name__ == "__main__":
    main()
