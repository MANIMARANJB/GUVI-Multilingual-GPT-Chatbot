from src.translation import (
    translate_to_english,
    translate_from_english
)

from src.retrieval import get_relevant_context
from src.generation import generate_response


# CHATBOT PIPELINE

def chatbot_response(user_input):
    """
    Process user input through translation,
    retrieval, generation, and reverse translation.
    """

    # TRANSLATE USER INPUT TO ENGLISH

    english_query, detected_language = translate_to_english(
        user_input
    )

    # RETRIEVE RELEVANT GUVI CONTEXT

    context = get_relevant_context(
        english_query,
        top_k=3
    )

    # GENERATE ENGLISH RESPONSE

    english_response = generate_response(
        english_query,
        context
    )

    # TRANSLATE RESPONSE BACK TO USER LANGUAGE

    final_response = translate_from_english(
        english_response,
        detected_language
    )

    return {
        "detected_language": detected_language,
        "english_query": english_query,
        "context": context,
        "english_response": english_response,
        "final_response": final_response
    }