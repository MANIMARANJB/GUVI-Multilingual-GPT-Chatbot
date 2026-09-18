import os

import faiss
import pandas as pd

from sentence_transformers import SentenceTransformer


# FILE PATHS

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CHUNKS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "guvi_chunks_with_embeddings_text.csv"
)

FAISS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "guvi_faiss.index"
)


# EMBEDDING MODEL

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# LOAD DATA

chunks_df = pd.read_csv(CHUNKS_PATH)


# LOAD FAISS INDEX

index = faiss.read_index(FAISS_PATH)


# LOAD EMBEDDING MODEL

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# SEARCH GUVI CONTENT

def search_guvi(query, top_k=5, candidate_k=40):
    """Search GUVI content using semantic similarity."""

    query_embedding = embedding_model.encode(
        [query]
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        candidate_k
    )

    results = chunks_df.iloc[
        indices[0]
    ].copy()

    results["score"] = scores[0]

    query_words = {
        word.lower()
        for word in query.split()
        if len(word) > 3
    }

    # TITLE BONUS

    def title_bonus(title):
        title = str(title).lower()

        matches = sum(
            word in title
            for word in query_words
        )

        return matches * 0.05

    results["title_bonus"] = (
        results["title"].apply(title_bonus)
    )

    # COURSE QUERY CHECK

    course_query = (
        "course" in query.lower()
        or "courses" in query.lower()
    )

    # COURSE LIST SCORE

    def course_list_score(text):
        if not course_query:
            return 0

        text = str(text)

        enrolled_count = text.count("Enrolled")
        hours_count = text.count("Hrs")

        return min(
            enrolled_count + hours_count,
            10
        ) * 0.05

    results["course_bonus"] = (
        results["chunk_text"].apply(
            course_list_score
        )
    )

    # NAVIGATION PENALTY

    def navigation_penalty(text):
        text = str(text)

        navigation_terms = [
            "Leaderboard",
            "Referral",
            "Rewards",
            "Profile",
            "Request a Callback",
            "That's It!",
            "Explore More"
        ]

        matches = sum(
            text.count(term)
            for term in navigation_terms
        )

        return matches * 0.08

    results["navigation_penalty"] = (
        results["chunk_text"].apply(
            navigation_penalty
        )
    )

    # FINAL SCORE

    results["final_score"] = (
        results["score"]
        + results["title_bonus"]
        + results["course_bonus"]
        - results["navigation_penalty"]
    )

    # REMOVE DUPLICATE CHUNKS

    results = results.drop_duplicates(
        subset=["chunk_text"]
    )

    results = results.sort_values(
        "final_score",
        ascending=False
    ).head(top_k)

    return results[
        [
            "title",
            "category",
            "chunk_text",
            "score",
            "final_score"
        ]
    ]


# BUILD CONTEXT

def get_relevant_context(query, top_k=5):
    """Retrieve relevant GUVI text for the chatbot."""

    results = search_guvi(
        query,
        top_k=top_k,
        candidate_k=40
    )

    context_parts = []

    for _, row in results.iterrows():
        context_parts.append(
            f"Title: {row['title']}\n"
            f"Category: {row['category']}\n"
            f"Content: {row['chunk_text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    return context