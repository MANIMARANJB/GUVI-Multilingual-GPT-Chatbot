import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from models.llm import generate_response

from models.language_detector import detect_language

from models.translator import (
    translate_to_english,
    translate_from_english
)


# Load embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# Load FAISS index
index = faiss.read_index(
    "embeddings/faiss_index/guvi.index"
)


# Load knowledge data
with open(
    "embeddings/faiss_index/knowledge.pkl",
    "rb"
) as f:
    knowledge = pickle.load(f)



def retrieve_context(question, top_k=2):

    query_embedding = embedding_model.encode(
        [question]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")


    distances, indices = index.search(
        query_embedding,
        top_k
    )


    context = ""


    for idx in indices[0]:

        item = knowledge[idx]

        context += (
            f"Topic: {item['topic']}\n"
            f"Question: {item['instruction']}\n"
            f"Answer: {item['response']}\n\n"
        )


    return context



def chat(question):

    # 1. Detect user language
    user_language = detect_language(
        question
    )

    print("Detected Language:", user_language)


    # 2. Translate input to English
    english_question = translate_to_english(
        question,
        user_language
    )
    print("English Question:", english_question)

    # 3. Retrieve knowledge using English question
    context = retrieve_context(
        english_question
    )


    # 4. Generate English answer
    english_answer = generate_response(
        english_question,
        context
    )


    # 5. Translate answer back
    final_answer = translate_from_english(
        english_answer,
        user_language
    )

    print("Output Language:", user_language)

    return final_answer
