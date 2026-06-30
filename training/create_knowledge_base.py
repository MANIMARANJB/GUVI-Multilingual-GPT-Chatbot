"""
GUVI Knowledge Base Creation

Converts scraped GUVI data into
structured chatbot knowledge.
"""

import json
import os
import re


INPUT_FILE = "data/raw/guvi_raw.json"

OUTPUT_FILE = "data/guvi_facts.json"


# ----------------------------------
# Load scraped data
# ----------------------------------

def load_data(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)



# ----------------------------------
# Clean text
# ----------------------------------

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def extract_course_info(text):

    keywords = [
        "Full Stack",
        "Data Science",
        "DevOps",
        "UI/UX",
        "Artificial Intelligence",
        "Machine Learning",
        "Python",
        "Java",
        "SQL",
        "self-paced",
        "live"
    ]


    sentences = text.split(".")


    useful_sentences = []


    for sentence in sentences:

        for keyword in keywords:

            if keyword.lower() in sentence.lower():

                useful_sentences.append(
                    sentence.strip()
                )

                break


    response = ". ".join(
        useful_sentences
    )


    return response[:600]

# ----------------------------------
# Create knowledge samples
# ----------------------------------

def create_facts(data):

    facts = []


    for item in data:

        content = clean_text(
            item.get(
                "content",
                ""
            )
        )


        category = item.get(
            "category",
            ""
        )


        if len(content) < 200:
            continue



        if category == "about":

            facts.append(
                {
                    "topic": "about_guvi",
                    "instruction": "What is GUVI?",
                    "response": extract_course_info(content)
                }
            )



        elif category == "courses":

            facts.append(
                {
                    "topic": "courses",
                    "instruction": "What courses are available in GUVI?",
                    "response": extract_course_info(content)
                }
            )



        elif category == "blog":

            title = item.get(
                "title",
                "GUVI blog"
            )


            facts.append(
                {
                    "topic": "learning_resources",
                    "instruction": f"Tell me about {title}",
                    "response": content[:500]
                }
            )


    return facts



# ----------------------------------
# Save knowledge base
# ----------------------------------

def save_data(data):

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
            data,
            file,
            indent=4,
            ensure_ascii=False
        )



# ----------------------------------
# Main
# ----------------------------------

def main():

    print(
        "Loading scraped data..."
    )


    data = load_data(
        INPUT_FILE
    )


    print(
        "Documents:",
        len(data)
    )


    facts = create_facts(
        data
    )


    save_data(
        facts
    )


    print(
        "Knowledge base created"
    )


    print(
        "Total facts:",
        len(facts)
    )



if __name__ == "__main__":

    main()