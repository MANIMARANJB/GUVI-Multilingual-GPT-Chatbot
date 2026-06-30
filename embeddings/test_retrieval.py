import pickle
import faiss
from sentence_transformers import SentenceTransformer


# -----------------------------------------
# Paths
# -----------------------------------------

INDEX_PATH = "embeddings/faiss_index/guvi.index"

DATA_PATH = "embeddings/faiss_index/knowledge.pkl"


# -----------------------------------------
# Load FAISS Index
# -----------------------------------------

print("Loading FAISS index...")

index = faiss.read_index(INDEX_PATH)


# -----------------------------------------
# Load Knowledge Data
# -----------------------------------------

print("Loading knowledge data...")

with open(DATA_PATH, "rb") as file:
    knowledge_data = pickle.load(file)


print("Total knowledge items:", len(knowledge_data))


# -----------------------------------------
# Load Embedding Model
# -----------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------------------
# User Query
# -----------------------------------------

query = input("\nAsk GUVI chatbot: ")


# -----------------------------------------
# Convert Query to Embedding
# -----------------------------------------

query_embedding = model.encode(
    [query],
    convert_to_numpy=True
)


# -----------------------------------------
# Search FAISS
# -----------------------------------------

k = 3

distances, indices = index.search(
    query_embedding,
    k
)


# -----------------------------------------
# Display Results
# -----------------------------------------

print("\nTop matching results:\n")


for i, idx in enumerate(indices[0]):

    item = knowledge_data[idx]

    print("--------------------------------")
    print("Match:", i + 1)
    print("Topic:", item["topic"])
    print("Question:", item["instruction"])
    print("Answer:", item["response"])
    print("Distance:", distances[0][i])