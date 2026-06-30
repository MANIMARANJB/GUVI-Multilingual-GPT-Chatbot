"""
GUVI Multilingual GPT Chatbot

Instruction Dataset Generator

Converts cleaned GUVI content into
small instruction-response training samples.
"""


import json
import os
import re


# ---------------------------------------
# Paths
# ---------------------------------------

INPUT_FILE = "data/processed/guvi_clean.json"

OUTPUT_FILE = "data/guvi_instruction_dataset.json"



# ---------------------------------------
# Load data
# ---------------------------------------

def load_data(path):
    """
    Load cleaned JSON data.
    """

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)



# ---------------------------------------
# Clean response text
# ---------------------------------------

def clean_response(text):
    """
    Remove website noise.
    """

    # remove repeated spaces

    text = re.sub(
        r"\s+",
        " ",
        text
    )


    # remove common UI words

    unwanted = [
        "login",
        "sign up",
        "signup",
        "apply now",
        "explore more",
        "learn more",
        "menu"
    ]


    for word in unwanted:

        text = re.sub(
            word,
            "",
            text,
            flags=re.IGNORECASE
        )


    return text.strip()



# ---------------------------------------
# Split text into chunks
# ---------------------------------------

def create_chunks(text, size=300):
    """
    Split long text into smaller
    training responses.

    Args:
        text: webpage content
        size: maximum words

    Returns:
        list of chunks
    """

    words = text.split()


    chunks = []


    for i in range(
        0,
        len(words),
        size
    ):

        chunk = " ".join(
            words[i:i+size]
        )


        if len(chunk) > 80:

            chunks.append(chunk)


    return chunks



# ---------------------------------------
# Generate questions
# ---------------------------------------

def generate_instruction(category):
    """
    Create user questions.
    """

    questions = {

        "about":
            [
                "What is GUVI?",
                "Tell me about GUVI platform."
            ],

        "courses":
            [
                "What courses does GUVI provide?",
                "Which learning programs are available in GUVI?"
            ],

        "blogs":
            [
                "What learning resources are available in GUVI?",
                "Tell me about GUVI blogs."
            ]

    }


    return questions.get(
        category,
        [
            "Tell me more about GUVI."
        ]
    )



# ---------------------------------------
# Create dataset
# ---------------------------------------

def create_dataset():

    print(
        "Loading cleaned data..."
    )


    data = load_data(
        INPUT_FILE
    )


    dataset = []


    for item in data:


        content = clean_response(
            item["content"]
        )


        chunks = create_chunks(
            content
        )


        questions = generate_instruction(
            item["category"]
        )


        for chunk in chunks:


            for question in questions:


                dataset.append(
                    {
                        "instruction": question,
                        "response": chunk
                    }
                )



    os.makedirs(
        "data",
        exist_ok=True
    )


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:


        json.dump(
            dataset,
            file,
            indent=4,
            ensure_ascii=False
        )


    print(
        "Instruction dataset created"
    )


    print(
        f"Total samples: {len(dataset)}"
    )



# ---------------------------------------
# Run
# ---------------------------------------

if __name__ == "__main__":

    create_dataset()