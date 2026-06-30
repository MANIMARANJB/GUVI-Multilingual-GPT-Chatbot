from models.llm import generate_response


answer = generate_response(
    "What is GUVI?",
    """
    GUVI is a multilingual technology learning platform.
    It provides courses in programming, AI, Data Science,
    and other technology skills.
    """
)


print(answer)