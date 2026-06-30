from models.rag_pipeline import chat


questions = [
    "What courses are available in GUVI?",
    "GUVI பற்றி சொல்லுங்கள்",
    "GUVI के बारे में बताएं"
]


for q in questions:

    print("\nQuestion:")
    print(q)

    print("\nAnswer:")

    answer = chat(q)

    print(answer)

    print("--------------------")
