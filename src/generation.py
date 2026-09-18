from transformers import AutoTokenizer, AutoModelForCausalLM


# GENERATION MODEL

GENERATION_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# LOAD TOKENIZER

tokenizer = AutoTokenizer.from_pretrained(
    GENERATION_MODEL_NAME
)


# LOAD MODEL

generation_model = AutoModelForCausalLM.from_pretrained(
    GENERATION_MODEL_NAME
)


# GENERATE RESPONSE

def generate_response(question, context):
    """Generate a grounded answer using retrieved GUVI content."""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a GUVI learning assistant. "
                "Answer only using information clearly present "
                "in the provided GUVI context. "
                "Do not invent course names, prices, ratings, "
                "durations, features, or other details. "
                "Ignore navigation text, ratings, enrollment counts, "
                "buttons, labels, and fragments such as 'View all courses'. "
                "If the user asks for courses, list only clear course "
                "or program names that appear in the context. "
                "Do not treat words such as English, hours, ratings, "
                "prices, or enrollment numbers as course names. "
                "Remove duplicates. "
                "If the context does not contain a complete list, "
                "say that these are some courses found in the available "
                "GUVI content rather than claiming they are all courses. "
                "Keep the answer clear and concise."
            )
        },
        {
            "role": "user",
            "content": (
                f"GUVI Context:\n{context}\n\n"
                f"User Question:\n{question}"
            )
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    output = generation_model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False
    )

    generated_tokens = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response.strip()