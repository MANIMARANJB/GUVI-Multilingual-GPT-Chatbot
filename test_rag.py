from models.rag_pipeline import chat


question = "What courses are available in GUVI?"


answer = chat(question)


print("\nQuestion:")
print(question)


print("\nAnswer:")
print(answer)
