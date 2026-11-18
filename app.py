
import streamlit as st
import pandas as pd
import requests
import os

from pathlib import Path
from dotenv import load_dotenv
from model.recommender import recommend_movies


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("TMDB_API_KEY")

if not API_KEY:
    try:
        API_KEY = st.secrets["TMDB_API_KEY"]
    except Exception:
        API_KEY = None

st.set_page_config(
    page_title="CineMatch | Discover Movies",
    page_icon="🎬",
    layout="wide"
)


# ==========================================
# 2. LOAD DATA
# ==========================================

movies = pd.read_csv(
    BASE_DIR / "data" / "processed_movies.csv"
)


# ==========================================
# 3. TMDB API
# ==========================================

@st.cache_data(ttl=3600)
def get_movie_details(movie_id):

    if not API_KEY:
        return None

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
# 4. CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background: #0b0b10;
    color: #ffffff;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

h1, h2, h3 {
    color: white;
}

.hero {
    padding: 35px;
    border-radius: 18px;
    background: linear-gradient(
        110deg,
        #25133d,
        #151522,
        #351326
    );
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 48px;
    margin-bottom: 5px;
}

.hero p {
    color: #c5c5d0;
    font-size: 17px;
}

.movie-title {
    font-size: 16px;
    font-weight: 700;
    color: white;
    margin-top: 8px;
}

.movie-meta {
    color: #aaaaaa;
    font-size: 13px;
}

.stButton > button {
    background-color: #e50914;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: bold;
    padding: 10px 22px;
}

.stButton > button:hover {
    background-color: #b80710;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 5. HERO SECTION
# ==========================================

st.markdown("""
<div class="hero">
    <h1>🎬 CineMatch</h1>
    <p>
        Discover your next favourite movie.
        Powered by Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 6. MOVIE SELECTION
# ==========================================

st.subheader("🔎 Find Your Next Movie")

selected_movie = st.selectbox(
    "Search for a movie",
    sorted(movies["title"].dropna().unique().tolist())
)

search_button = st.button(
    "✨ Get My Recommendations",
    type="primary"
)


# ==========================================
# 7. SELECTED MOVIE DETAILS
# ==========================================

if search_button:

    selected_row = movies[
        movies["title"] == selected_movie
    ].iloc[0]

    movie_id = int(selected_row["id"])

    details = get_movie_details(movie_id)

    st.divider()
    st.subheader("🎥 Your Selected Movie")

    if details:

        col1, col2 = st.columns([1, 3])

        with col1:

            poster = details.get("poster_path")

            if poster:
                st.image(
                    "https://image.tmdb.org/t/p/w500" + poster,
                    use_container_width=True
                )
            else:
                st.info("Poster unavailable")

        with col2:

            st.title(details.get("title", selected_movie))

            st.write(
                details.get("overview")
                or "No description available."
            )

            st.write(
                f"⭐ Rating: {details.get('vote_average', 'N/A')}/10"
            )

            st.write(
                f"📅 Release Date: {details.get('release_date', 'N/A')}"
            )

            st.write(
                f"⏱️ Runtime: {details.get('runtime', 'N/A')} minutes"
            )

            genres = [
                genre["name"]
                for genre in details.get("genres", [])
            ]

            st.write(
                "🎭 Genres: " + ", ".join(genres)
            )

    else:
        st.warning("Could not load movie details from TMDB.")

    
# ======================================
# 8. INTERACTIVE RECOMMENDATION CARDS
# ======================================

st.divider()

st.subheader("✨ Movies You Might Also Like")

recommendations = recommend_movies(selected_movie)

if recommendations:

    columns = st.columns(5)

    for i, movie in enumerate(recommendations):

        with columns[i]:

            movie_row = movies[
                movies["title"] == movie["title"]
            ].iloc[0]

            rec_id = int(movie_row["id"])

            rec_details = get_movie_details(rec_id)

            if rec_details:

                # Movie poster
                poster = rec_details.get("poster_path")

                if poster:
                    st.image(
                        "https://image.tmdb.org/t/p/w500" + poster,
                        use_container_width=True
                    )
                else:
                    st.info("Poster unavailable")

                # Movie title
                st.markdown(
                    f"**{rec_details.get('title', movie['title'])}**"
                )

                # Rating
                rating = rec_details.get("vote_average", "N/A")

                st.caption(f"⭐ Rating: {rating}/10")

                # Release year
                release_date = rec_details.get("release_date") or "N/A"

                st.caption(f"📅 {release_date[:4]}")

                # ML similarity
                st.caption(
                    f"🧠 Similarity: {movie['similarity']:.2%}"
                )

                # Expandable movie details
                with st.expander("View Details"):

                    st.write(
                        rec_details.get("overview")
                        or "No overview available."
                    )

                    genres = [
                        genre["name"]
                        for genre in rec_details.get("genres", [])
                    ]

                    st.write("🎭 Genres:", ", ".join(genres) or "N/A")

                    st.write(
                        "📅 Release Date:",
                        release_date
                    )

                    st.write(
                        "⏱️ Runtime:",
                        f"{rec_details.get('runtime', 'N/A')} minutes"
                    )

                    st.write(
                        "⭐ TMDB Rating:",
                        f"{rating}/10"
                    )

                    st.write(
                        "🧠 ML Similarity:",
                        f"{movie['similarity']:.2%}"
                    )

            else:
                st.write(movie["title"])
                st.warning("Movie details unavailable.")

else:
    st.warning("No recommendations found.")



# ==========================================
# 9. FOOTER
# ==========================================

st.divider()

st.caption(
    "CineMatch | Built with Python, Scikit-learn, "
    "Streamlit and TMDB API. "
    "Built by Yeamin"
)
