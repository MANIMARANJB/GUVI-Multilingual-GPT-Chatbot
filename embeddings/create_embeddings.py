import json
import os
import pickle

import faiss
from sentence_transformers import SentenceTransformer


# -----------------------------------------
# Paths
# -----------------------------------------

KNOWLEDGE_FILE = "data/knowledge/guvi_knowledge.json"

INDEX_PATH = "embeddings/faiss_index/guvi.index"

DATA_PATH = "embeddings/faiss_index/knowledge.pkl"


# -----------------------------------------
# Load Knowledge Base
# -----------------------------------------

print("Loading knowledge base...")

with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as file:
    knowledge_data = json.load(file)


print("Total documents:", len(knowledge_data))


# -----------------------------------------
# Prepare Text Data
# -----------------------------------------

texts = []

for item in knowledge_data:

    text = (
        "Question: "
        + item["instruction"]
        + "\nAnswer: "
        + item["response"]
    )

    texts.append(text)


# -----------------------------------------
# Load Embedding Model
# -----------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------------------
# Create Embeddings
# -----------------------------------------

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    convert_to_numpy=True
)


print("Embedding shape:", embeddings.shape)


# -----------------------------------------
# Create FAISS Index
# -----------------------------------------

dimension = embeddings.shape[1]


index = faiss.IndexFlatL2(
    dimension
)


index.add(
    embeddings
)


# -----------------------------------------
# Save FAISS Index
# -----------------------------------------

faiss.write_index(
    index,
    INDEX_PATH
)


# -----------------------------------------
# Save Original Data
# -----------------------------------------

with open(DATA_PATH, "wb") as file:

    pickle.dump(
        knowledge_data,
        file
    )


print("FAISS index created successfully")
print("Stored vectors:", index.ntotal)