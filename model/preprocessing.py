
import pandas as pd
import ast
from pathlib import Path


# ==========================================
# STEP 1: PROJECT DIRECTORY
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]


# ==========================================
# STEP 2: LOAD DATASETS
# ==========================================

movies = pd.read_csv(
    BASE_DIR / "data" / "tmdb_5000_movies.csv"
)

credits = pd.read_csv(
    BASE_DIR / "data" / "tmdb_5000_credits.csv"
)

print("Datasets loaded successfully!")


# ==========================================
# STEP 3: MERGE DATASETS
# ==========================================

# Remove duplicate title column from credits
credits = credits.drop(columns=["title"])

# Merge using movie IDs
df = movies.merge(
    credits,
    left_on="id",
    right_on="movie_id",
    how="left",
    validate="one_to_one"
)

print("\nMERGED DATASET")
print("Shape:", df.shape)


# ==========================================
# STEP 4: HANDLE MISSING VALUES
# ==========================================

df["overview"] = df["overview"].fillna("")


# ==========================================
# STEP 5: FEATURE EXTRACTION FUNCTIONS
# ==========================================

# Convert string into Python objects
def convert(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError):
        return []


# Extract genres and keywords
def extract_names(text):
    items = convert(text)

    return [
        item["name"]
        for item in items
        if "name" in item
    ]


# Extract top 3 actors
def extract_cast(text):
    items = convert(text)

    return [
        item["name"]
        for item in items[:3]
        if "name" in item
    ]


# Extract director
def extract_director(text):
    items = convert(text)

    for item in items:
        if item.get("job") == "Director":
            return [item["name"]]

    return []


# ==========================================
# STEP 6: APPLY FEATURE EXTRACTION
# ==========================================

df["genres"] = df["genres"].apply(extract_names)

df["keywords"] = df["keywords"].apply(extract_names)

df["cast"] = df["cast"].apply(extract_cast)

df["crew"] = df["crew"].apply(extract_director)

df["overview"] = df["overview"].apply(
    lambda x: x.split()
)

print("\nFEATURE EXTRACTION COMPLETED!")


# ==========================================
# STEP 7: CREATE TAGS
# ==========================================

df["tags"] = (
    df["genres"]
    + df["keywords"]
    + df["cast"]
    + df["crew"]
    + df["overview"]
)

# Convert lists into strings
df["tags"] = df["tags"].apply(
    lambda x: " ".join(x)
)


# ==========================================
# STEP 8: CREATE FINAL DATASET
# ==========================================

final_df = df[["id", "title", "tags"]].copy()


# ==========================================
# STEP 9: SAVE PROCESSED DATASET
# ==========================================

final_df.to_csv(
    BASE_DIR / "data" / "processed_movies.csv",
    index=False
)


# ==========================================
# STEP 10: DISPLAY RESULTS
# ==========================================

print("\nEXTRACTED FEATURES")

print(
    df[[
        "title",
        "genres",
        "keywords",
        "cast",
        "crew",
        "overview"
    ]].head()
)

print("\nFINAL DATASET")

print(final_df.head())

print("\nFinal dataset shape:", final_df.shape)

print("\nProcessed dataset saved successfully!")
