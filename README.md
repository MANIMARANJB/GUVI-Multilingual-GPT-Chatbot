# GUVI Multilingual AI Chatbot

## Multilingual RAG-Based Learning Assistant using Translation, FAISS and Qwen

The **GUVI Multilingual AI Chatbot** is an NLP and Generative AI project designed to answer questions related to GUVI courses, programs, practice platforms, and learning resources.

The chatbot supports multiple Indian languages and uses a **Retrieval-Augmented Generation (RAG)** architecture.

When a user asks a question in a non-English language, the system automatically detects the language, translates the question into English, retrieves relevant information from a GUVI-specific knowledge base, generates an answer using an instruction-tuned language model, and translates the final response back into the user's original language.

The application is developed using **Python and Streamlit** and can run locally.

---

# Project Objective

The main objective of this project is to develop a multilingual AI assistant capable of:

- Understanding questions in multiple languages
- Automatically detecting the user's language
- Translating non-English queries into English
- Searching a GUVI-specific knowledge base
- Retrieving semantically relevant information
- Generating context-aware responses using an LLM
- Translating responses back to the user's language
- Providing an interactive chatbot interface using Streamlit

---

# Problem Statement

Educational platforms contain large amounts of information about courses, programs, certifications, practice platforms, and learning resources.

Finding the required information can sometimes be difficult, especially when users prefer interacting in their native language.

This project solves this problem by combining:

- Natural Language Processing
- Machine Translation
- Sentence Embeddings
- Semantic Search
- Vector Retrieval
- Retrieval-Augmented Generation
- Large Language Models
- Streamlit

The chatbot allows users to interact with GUVI-related information using their preferred supported language.

---

# Supported Languages

The current application supports:

- English
- Tamil
- Hindi
- Telugu
- Kannada
- Malayalam

English is used as the intermediate language for retrieval and response generation because the GUVI knowledge base primarily contains English content.

---

# Project Architecture

The project contains two major workflows:

1. Knowledge Base Creation
2. Chatbot Query Processing

## 1. Knowledge Base Creation

```text
GUVI Website
     |
     v
Web Scraping
Requests + BeautifulSoup
     |
     v
Raw GUVI Content
     |
     v
Text Cleaning
     |
     v
Text Chunking
200 words + 30 word overlap
     |
     v
Sentence Transformer
all-MiniLM-L6-v2
     |
     v
384-Dimensional Embeddings
     |
     v
FAISS Vector Index
     |
     v
GUVI Knowledge Base
```

## 2. Chatbot Query Processing

```text
                     USER QUESTION
                           |
                           v
                  Language Detection
                     langdetect
                           |
                           v
                  Is Input English?
                     /           \
                   No             Yes
                   |               |
                   v               |
             NLLB Translation      |
             to English            |
                   |               |
                   +-------+-------+
                           |
                           v
                  English Question
                           |
                           v
                 MiniLM Embedding
                           |
                           v
                   FAISS Search
                           |
                           v
             Relevant GUVI Chunks
                           |
                           v
                Context + Question
                           |
                           v
               Qwen2.5-0.5B-Instruct
                           |
                           v
                   English Answer
                           |
                           v
                Original Language?
                     /           \
                   Yes            No
                   |               |
                   v               |
             NLLB Translation      |
                   |               |
                   +-------+-------+
                           |
                           v
                     FINAL ANSWER
                           |
                           v
                     STREAMLIT UI
```

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Requests | Download GUVI webpages |
| BeautifulSoup | Web scraping and HTML parsing |
| Pandas | Data manipulation |
| Regular Expressions | Text cleaning |
| Sentence Transformers | Generate semantic embeddings |
| MiniLM | Embedding model |
| FAISS | Vector similarity search |
| langdetect | Language detection |
| NLLB | Multilingual translation |
| Hugging Face Transformers | Loading transformer models |
| Qwen2.5 | Response generation |
| PyTorch | Deep learning backend |
| Streamlit | Chatbot user interface |

---

# Models Used

The project uses three main pretrained model components.

## 1. Translation Model

### Model

`facebook/nllb-200-distilled-600M`

### Purpose

The NLLB model is responsible for multilingual translation.

NLLB stands for:

**No Language Left Behind**

The translation workflow is:

```text
Tamil / Hindi / Telugu / Kannada / Malayalam
                     |
                     v
                   NLLB
                     |
                     v
                  English
```

After the chatbot generates an English response:

```text
English Response
       |
       v
      NLLB
       |
       v
User's Original Language
```

This allows the retrieval and generation pipeline to operate mainly in English while still supporting multilingual conversations.

---

# 2. Embedding Model

### Model

`sentence-transformers/all-MiniLM-L6-v2`

### Purpose

MiniLM converts textual information into numerical vector representations called **embeddings**.

Each text chunk is converted into a:

```text
384-dimensional vector
```

Example:

```text
"Python for Data Science"

        |
        v

MiniLM Embedding Model

        |
        v

[0.21, -0.34, 0.76, ..., 0.18]
```

Semantically similar text generally produces vectors that are closer together.

This enables semantic search.

---

# 3. Response Generation Model

### Model

`Qwen/Qwen2.5-0.5B-Instruct`

### Purpose

Qwen is the main response-generation model.

The model receives:

```text
User Question
      +
Retrieved GUVI Context
```

and generates a natural-language response.

A lightweight 0.5B instruction model was selected so that the application can be developed and tested locally without requiring very large computational resources.

---

# Additional Components

## Language Detection

The project uses:

`langdetect`

It identifies the user's language before translation.

Example:

```text
Input:

எனக்கு டேட்டா சயின்ஸ் பற்றி தகவல் வேண்டும்

Detected Language:

ta
```

Language codes used by the application include:

```text
en -> English
ta -> Tamil
hi -> Hindi
te -> Telugu
kn -> Kannada
ml -> Malayalam
```

---

# FAISS Vector Search

FAISS is used to perform fast similarity search between the user's question and GUVI content.

The project uses:

```python
faiss.IndexFlatIP
```

Embeddings are L2-normalized before indexing.

```python
faiss.normalize_L2(embeddings)
```

With normalized embeddings, inner-product similarity can be used to rank vectors similarly to cosine similarity.

The final knowledge base currently contains approximately:

```text
518 text chunks
518 embeddings
518 FAISS vectors
```

Each embedding contains:

```text
384 dimensions
```

---

# What is RAG?

RAG stands for:

**Retrieval-Augmented Generation**

A traditional LLM application works approximately like this:

```text
Question
   |
   v
LLM
   |
   v
Answer
```

The application developed in this project works differently:

```text
Question
   |
   v
Search GUVI Knowledge Base
   |
   v
Retrieve Relevant Information
   |
   v
Question + Retrieved Context
   |
   v
LLM
   |
   v
Answer
```

This approach helps the language model generate responses using relevant GUVI information instead of relying entirely on its pretrained knowledge.

---

# Data Collection

GUVI website content was collected using:

- Requests
- BeautifulSoup

Relevant information was extracted from webpages, including:

```text
URL
Title
Category
Content
```

Example structure:

```text
URL:
https://www.guvi.in/...

Title:
GUVI Course Page

Category:
Course

Content:
Extracted webpage information...
```

The final cleaned raw dataset contains approximately:

```text
45 GUVI webpages
```

The raw dataset is stored in:

```text
data/guvi_raw_data.csv
```

---

# Data Preprocessing

The collected website content contains formatting, unnecessary whitespace, and other webpage-related noise.

Text preprocessing includes:

- Converting data to string format
- Removing repeated whitespace
- Cleaning punctuation spacing
- Removing unnecessary separators
- Removing very small chunks
- Removing duplicates where required

Example:

```python
def preprocess_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"[|]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()
```

---

# Text Chunking

Long webpages are not directly used for retrieval.

Instead, webpage content is divided into smaller chunks.

The project uses approximately:

```text
Chunk Size : 200 words
Overlap    : 30 words
```

Example:

```text
Original Document
        |
        v
+-----------------------+
| Chunk 1: Words 1-200  |
+-----------------------+
           |
           v
+-------------------------+
| Chunk 2: Words 171-370 |
+-------------------------+
           |
           v
+-------------------------+
| Chunk 3: Words 341-540 |
+-------------------------+
```

The overlap helps preserve information that may occur near chunk boundaries.

Chunks containing fewer than approximately 50 words are removed.

The final processed dataset contains approximately:

```text
518 chunks
```

---

# Semantic Search

Semantic search searches according to the **meaning of the query**, rather than relying only on exact keyword matching.

For example:

```text
User Query:

"Are there any artificial intelligence courses?"
```

A semantic search system may retrieve information containing:

```text
Introduction to Generative AI
Machine Learning
LLM
AI Programs
```

even when the exact phrase used by the user is not present.

The workflow is:

```text
User Question
      |
      v
MiniLM
      |
      v
Query Embedding
      |
      v
FAISS
      |
      v
Compare with Stored Embeddings
      |
      v
Most Relevant GUVI Chunks
```

---

# Retrieval Improvement

During testing, some retrieved chunks contained repeated website navigation information such as:

```text
Leaderboard
Referral
Rewards
Profile
Explore More
```

These chunks could reduce response quality.

The retrieval system was therefore improved using additional ranking logic.

The final ranking considers:

```text
Semantic Similarity
        +
Title Relevance
        +
Course Content Relevance
        -
Navigation Content Penalty
        |
        v
Final Retrieval Score
```

This helps prioritize useful GUVI information.

---

# Complete Example

Suppose a user asks a question in Tamil:

```text
எனக்கு டேட்டா சயின்ஸ் பாடநெறி பற்றி தகவல் வேண்டும்
```

## Step 1 - Language Detection

`langdetect` detects:

```text
ta
```

which represents Tamil.

## Step 2 - Translation

NLLB translates the question to English.

```text
I want information about the Data Science course.
```

## Step 3 - Query Embedding

MiniLM converts the English question into a:

```text
384-dimensional vector
```

## Step 4 - Semantic Retrieval

FAISS compares the query embedding against the stored GUVI vectors.

```text
Query Vector
      |
      v
FAISS
      |
      v
518 Stored Vectors
      |
      v
Top Relevant Chunks
```

Relevant Data Science information is retrieved.

## Step 5 - Context Creation

The retrieved chunks are combined into context.

Example:

```text
Title: Data Science Course With Placement

Category: Live Program

Content:
Relevant GUVI Data Science information...
```

## Step 6 - Response Generation

The question and retrieved context are passed to:

```text
Qwen2.5-0.5B-Instruct
```

The model generates an English answer based on the retrieved information.

## Step 7 - Reverse Translation

Because the original language was Tamil, NLLB translates the English answer back to Tamil.

## Step 8 - Streamlit

The translated response is displayed to the user through the Streamlit chatbot interface.

Final workflow:

```text
Tamil Question
      |
      v
Language Detection
      |
      v
English Translation
      |
      v
Semantic Retrieval
      |
      v
GUVI Context
      |
      v
Qwen
      |
      v
English Response
      |
      v
Tamil Translation
      |
      v
Streamlit
```

---

# Project Folder Structure

```text
guvi_multilingual_chatbot/
│
├── data/
│   ├── guvi_raw_data.csv
│   ├── guvi_chunks.csv
│   ├── guvi_chunks_with_embeddings_text.csv
│   └── guvi_faiss.index
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_embeddings_faiss.ipynb
│   └── 04_translation.ipynb
│
├── src/
│   ├── translation.py
│   ├── retrieval.py
│   ├── generation.py
│   └── chatbot.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

# File Explanation

## `01_data_collection.ipynb`

Responsible for:

- Discovering GUVI URLs
- Scraping GUVI webpages
- Extracting webpage content
- Categorizing webpages
- Creating the raw dataset

Output:

```text
guvi_raw_data.csv
```

---

## `02_data_preprocessing.ipynb`

Responsible for:

- Cleaning scraped text
- Removing unnecessary formatting
- Splitting documents into chunks
- Adding chunk IDs
- Removing very small chunks

Output:

```text
guvi_chunks.csv
```

---

## `03_embeddings_faiss.ipynb`

Responsible for:

- Loading MiniLM
- Generating embeddings
- Normalizing embeddings
- Creating the FAISS index
- Saving the vector index

Outputs:

```text
guvi_chunks_with_embeddings_text.csv
guvi_faiss.index
```

---

## `04_translation.ipynb`

Used for:

- Testing language detection
- Testing NLLB
- Tamil-to-English translation
- English-to-Tamil translation
- Testing multilingual translation workflow

---

## `src/translation.py`

Responsible for:

```text
Language Detection
        |
        v
Translation to English
        |
        v
Reverse Translation
```

---

## `src/retrieval.py`

Responsible for:

```text
Load MiniLM
      |
      v
Load FAISS
      |
      v
Convert Query to Embedding
      |
      v
Semantic Search
      |
      v
Reranking
      |
      v
Relevant GUVI Context
```

---

## `src/generation.py`

Responsible for loading Qwen and generating an answer using:

```text
User Question
      +
Retrieved Context
      |
      v
Qwen2.5
      |
      v
Generated Response
```

---

## `src/chatbot.py`

Acts as the main pipeline controller.

It connects:

```text
translation.py
      |
      v
retrieval.py
      |
      v
generation.py
      |
      v
translation.py
```

The main chatbot workflow is controlled through:

```python
chatbot_response(user_input)
```

---

## `app.py`

Provides the Streamlit user interface.

It handles:

- User input
- Chat messages
- Session history
- Chatbot pipeline execution
- Response display
- Error handling
- Clear conversation option

---

# Installation

Clone the repository:

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd guvi_multilingual_chatbot
```

Create a virtual environment if required:

```bash
python -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

# Run the Application

Run Streamlit from the project root:

```bash
streamlit run app.py
```

Streamlit will start the chatbot locally.

The application can normally be accessed through the local Streamlit address displayed in the terminal.

---

# Key Features

- Multilingual chatbot
- Automatic language detection
- Translation to and from English
- GUVI-specific knowledge base
- Web scraping
- Text preprocessing
- Text chunking
- Sentence embeddings
- Semantic search
- FAISS vector retrieval
- Retrieval reranking
- Retrieval-Augmented Generation
- Lightweight local LLM
- Interactive Streamlit interface
- Modular Python architecture
- Error handling
- Conversation history

---

# Challenges Faced

## 1. Webpage Noise

GUVI webpages contained navigation information and repeated content.

### Solution

Text cleaning and retrieval penalties were introduced to reduce the importance of navigation-heavy chunks.

---

## 2. Incomplete Course Retrieval

Initial searches did not always retrieve the most relevant GUVI course information.

### Solution

Additional GUVI course pages were collected and the knowledge base was rebuilt.

---

## 3. Retrieval Ranking

Semantic similarity alone sometimes returned irrelevant content.

### Solution

The retrieval ranking was improved using:

- Semantic similarity
- Title relevance
- Course-content bonus
- Navigation penalty

---

## 4. Multilingual Retrieval

The knowledge base mainly contains English information while users may ask questions in different languages.

### Solution

Non-English queries are translated into English before semantic retrieval.

---

# Limitations

The current project has several limitations:

- Response quality depends on the available GUVI website content.
- Dynamically loaded website information may not be captured by basic web scraping.
- Translation may occasionally lose context.
- A lightweight LLM has lower generation capability than much larger models.
- Retrieval quality depends on the quality of chunks and embeddings.
- RAG reduces hallucination but cannot completely eliminate it.
- The current knowledge base needs to be rebuilt when source content changes significantly.

---

# Future Improvements

Future versions can include:

- More supported languages
- Automated knowledge-base updates
- Better webpage boilerplate removal
- Improved reranking models
- Structured GUVI course catalog retrieval
- Larger or domain-adapted LLM
- Model fine-tuning using GUVI-specific question-answer data
- Retrieval evaluation using a dedicated test dataset
- User authentication
- Feedback collection
- Chat-history database
- Cloud deployment
- API-based architecture

---

# Why This Project Uses RAG Instead of Only an LLM

A standalone LLM depends mainly on information learned during model training.

This can result in:

- Outdated information
- Missing GUVI-specific information
- Hallucinated answers

RAG provides relevant GUVI information to the model at query time.

```text
Standalone LLM

Question
   |
   v
LLM Knowledge
   |
   v
Answer


RAG

Question
   |
   v
GUVI Knowledge Base
   |
   v
Relevant Context
   |
   v
LLM
   |
   v
Grounded Answer
```

Therefore, RAG is more suitable for this domain-specific chatbot.

---

# Model Summary

| Component | Technology / Model | Role |
|---|---|---|
| Language Detection | langdetect | Detect input language |
| Translation | facebook/nllb-200-distilled-600M | Translate user queries and responses |
| Embeddings | all-MiniLM-L6-v2 | Generate semantic vectors |
| Vector Search | FAISS IndexFlatIP | Retrieve relevant GUVI chunks |
| Generation | Qwen2.5-0.5B-Instruct | Generate context-aware answers |
| Frontend | Streamlit | Provide chatbot interface |

---

# Final Project Pipeline

```text
DATA PIPELINE

GUVI Website
     ↓
Web Scraping
     ↓
Raw Dataset
     ↓
Text Cleaning
     ↓
Chunking
     ↓
MiniLM Embeddings
     ↓
FAISS Index
     ↓
GUVI Knowledge Base


CHATBOT PIPELINE

User
 ↓
Streamlit
 ↓
Language Detection
 ↓
NLLB Translation
 ↓
MiniLM Query Embedding
 ↓
FAISS Semantic Search
 ↓
Relevant GUVI Context
 ↓
Qwen2.5
 ↓
English Response
 ↓
NLLB Reverse Translation
 ↓
Streamlit
 ↓
User
```

---

# Conclusion

The GUVI Multilingual AI Chatbot demonstrates how **Natural Language Processing, machine translation, sentence embeddings, vector search, Retrieval-Augmented Generation, Large Language Models, and Streamlit** can be integrated into a single real-world application.

Instead of depending completely on the pretrained knowledge of an LLM, the application retrieves relevant information from a GUVI-specific knowledge base before generating a response.

The multilingual architecture further allows users to interact with the system in their preferred supported language.

The project demonstrates an end-to-end AI workflow covering:

```text
Data Collection
      ↓
Data Cleaning
      ↓
NLP Preprocessing
      ↓
Embeddings
      ↓
Vector Database
      ↓
Semantic Retrieval
      ↓
RAG
      ↓
LLM Generation
      ↓
Translation
      ↓
Streamlit Application
```

---

# Author

**Manimaran JB**

Data Science Project

GUVI
