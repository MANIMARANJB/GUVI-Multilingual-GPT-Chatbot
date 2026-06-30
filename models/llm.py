import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# Load .env file
load_dotenv()


# Get Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")


# Create Hugging Face client
client = InferenceClient(
    token=HF_TOKEN
)


def generate_response(question, context):

    prompt = f"""
You are GUVI AI Assistant.

Answer the user question using only the information provided in the context.

If the answer is not available in the context, say:
"I don't have enough information about this topic."

Do not create false information.

Context:
{context}


User Question:
{question}


Answer:
"""


    try:

        response = client.chat_completion(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=300,
            temperature=0.3
        )


        answer = response.choices[0].message.content

        return answer


    except Exception as e:

        return f"LLM Error: {str(e)}"
