
import streamlit as st
import pandas as pd
import requests
import os

from pathlib import Path
from dotenv import load_dotenv
from model.recommender import recommend_movies

# ==========================================
# STEP 1: CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("TMDB_API_KEY")

st.set_page_config(
    page_title="CineMatch | Movie Recommendations",
    page_icon="🎬",
    layout="wide"
)

# ==========================================
# STEP 2: LOAD MOVIE DATA
# ==========================================

movies = pd.read_csv(
    BASE_DIR / "data" / "processed_movies.csv"
)

# ==========================================
# STEP 3: TMDB API FUNCTION
# ==========================================

@st.cache_data(ttl=3600)
def get_movie_details(movie_id):

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"

    try:
        response = requests.get(
            url,
            params={"api_key": API_KEY},
            timeout=15
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException:
        return None

# ==========================================
# STEP 4: CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #101014;
    color: white;
}

h1, h2, h3 {
    color: white;
}

.movie-card {
    background-color: #1c1c24;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 15px;
}

.movie-title {
    font-size: 17px;
    font-weight: bold;
    color: white;
}

.movie-info {
    color: #aaaaaa;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# STEP 5: HEADER
# ==========================================

st.title("🎬 CineMatch")

st.markdown(
    "### Discover movies you'll love."
)

st.caption(
    "An ML-powered movie recommendation system."
)

st.divider()

# ==========================================
# STEP 6: MOVIE SEARCH
# ==========================================

st.subheader("Find Your Next Movie")

selected_movie = st.selectbox(
    "Search or select a movie",
    movies["title"].dropna().tolist()
)

if st.button("Get Recommendations", type="primary"):

    selected_row = movies[
        movies["title"] == selected_movie
    ].iloc[0]

    movie_id = int(selected_row["id"])

    # Fetch selected movie details
    details = get_movie_details(movie_id)

    st.divider()

    st.subheader("Your Selected Movie")

    if details:

        col1, col2 = st.columns([1, 3])

        with col1:
            if details.get("poster_path"):
                st.image(
                    "https://image.tmdb.org/t/p/w500"
                    + details["poster_path"]
                )

        with col2:

            st.header(details.get("title", selected_movie))

            st.write(
                details.get("overview")
                or "No description available."
            )

            st.write(
                "⭐ Rating:",
                details.get("vote_average", "N/A")
            )

            st.write(
                "📅 Release Date:",
                details.get("release_date", "N/A")
            )

            genres = [
                genre["name"]
                for genre in details.get("genres", [])
            ]

            st.write("🎭 Genres:", ", ".join(genres))

    else:
        st.warning(
            "Could not load movie details from TMDB."
        )

    # ======================================
    # STEP 7: GENERATE RECOMMENDATIONS
    # ======================================

    st.divider()

    st.subheader("✨ Recommended Movies For You")

    recommendations = recommend_movies(selected_movie)

    if recommendations:

        columns = st.columns(5)

        for i, movie in enumerate(recommendations):

            with columns[i]:

                # Find recommended movie ID
                movie_row = movies[
                    movies["title"] == movie["title"]
                ].iloc[0]

                rec_id = int(movie_row["id"])

                # Fetch TMDB information
                rec_details = get_movie_details(rec_id)

                if rec_details:

                    poster = rec_details.get("poster_path")

                    if poster:
                        st.image(
                            "https://image.tmdb.org/t/p/w500"
                            + poster,
                            use_container_width=True
                        )

                    st.markdown(
                        f"**{rec_details.get('title', movie['title'])}**"
                    )

                    st.caption(
                        f"⭐ {rec_details.get('vote_average', 'N/A')} "
                        f"| {rec_details.get('release_date', 'N/A')[:4]}"
                    )

                    st.caption(
                        f"Similarity: {movie['similarity']:.2%}"
                    )

                else:
                    st.write(movie["title"])

    else:
        st.warning("No recommendations found.")

# ==========================================
# STEP 8: FOOTER
# ==========================================

st.divider()

st.caption(
    "CineMatch | Powered by Machine Learning "
    "and The Movie Database (TMDB). "
    "This product uses the TMDB API but is not endorsed "
    "or certified by TMDB."
)