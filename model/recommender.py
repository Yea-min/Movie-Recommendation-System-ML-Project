
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# STEP 1: LOAD PROCESSED DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

movies = pd.read_csv(
    BASE_DIR / "data" / "processed_movies.csv"
)

movies["tags"] = movies["tags"].fillna("").str.lower()

print("Dataset loaded successfully!")
print("Total movies:", len(movies))


# ==========================================
# STEP 2: TF-IDF VECTORIZATION
# ==========================================

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(movies["tags"])

print("\nTF-IDF completed!")
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)


# ==========================================
# STEP 3: RECOMMENDATION FUNCTION
# ==========================================

def recommend_movies(movie_title, number_of_recommendations=5):

    # Find the selected movie
    matching_movies = movies[
        movies["title"].str.casefold() == movie_title.casefold()
    ]

    if matching_movies.empty:
        print("Movie not found!")
        return []

    # Get the movie's index
    movie_index = matching_movies.index[0]

    # Calculate similarity with all movies
    similarity_scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    # Sort movies by similarity score
    similar_movies = sorted(
        enumerate(similarity_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Store recommendations
    recommendations = []

    for index, score in similar_movies:

        # Skip the selected movie itself
        if index == movie_index:
            continue

        recommendations.append({
            "title": movies.iloc[index]["title"],
            "similarity": round(float(score), 4)
        })

        if len(recommendations) == number_of_recommendations:
            break

    return recommendations


# ==========================================
# STEP 4: TEST OUR MODEL
# ==========================================

if __name__ == "__main__":

    selected_movie = "Interstellar"

    print("\nSelected Movie:", selected_movie)

    recommendations = recommend_movies(selected_movie)

    print("\nRecommended Movies:")

    for movie in recommendations:
        print(
            f"{movie['title']} "
            f"- Similarity: {movie['similarity']}"
        )
