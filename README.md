# 🎬 Movie Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_LINK_HERE)

A content-based movie recommendation web application built with **Streamlit**, **Scikit-Learn**, and **Pandas**, trained on the TMDB 5000 Movie Dataset. The app analyzes plot overviews, genres, keywords, top cast members, and directors to generate instant, personalized recommendations.

---

## 🚀 Live Demo

Check out the live interactive app: **[Launch Movie Recommender App](YOUR_STREAMLIT_APP_LINK_HERE)**

---

## ✨ Key Features

* **Multiple Recommendation Modes:**
  * **Combined (Recommended):** Blends plot summaries with cast, director, genres, and keywords.
  * **Plot & Themes Only:** Matches strictly on plot summary text using TF-IDF vectorization.
  * **Cast, Director & Genres:** Matches metadata soup using CountVectorizer.
  * **Hybrid Strategy:** Combines content similarity scores with IMDB weighted quality ratings to favor better-rated movies.
* **On-the-Fly Vectorization:** Computes matrix features efficiently in memory using Streamlit caching (`@st.cache_resource`), eliminating the need for heavy `.pkl` artifact storage.
* **Top-Rated Leaderboard:** Features a dedicated tab exploring top-rated movies scored via IMDB's weighted rating formula.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Data Processing & Analysis:** Pandas, NumPy
* **Machine Learning & NLP:** Scikit-Learn (`TfidfVectorizer`, `CountVectorizer`, `cosine_similarity`)
* **Web Framework:** Streamlit
* **Exploratory Data Analysis:** Jupyter Notebook (`Recommendation.ipynb`)

---

## 📁 Repository Structure

```text
.
├── Recommendation.ipynb     # Complete data preprocessing, EDA, and feature engineering notebook
├── recommendation_app.py   # Streamlit web app script with cached matrix computations
├── movies_for_app.csv       # Preprocessed dataset containing movie metadata & soup features
├── top_rated_for_app.csv    # Weighted ratings dataset for the leaderboard
└── requirements.txt         # Environment dependencies for deployment
