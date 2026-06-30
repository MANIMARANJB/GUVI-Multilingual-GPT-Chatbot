"""
GUVI Dataset Cleaning Pipeline
"""

import json
import os
import re


INPUT_FILE = "data/raw/guvi_raw.json"

OUTPUT_FILE = "data/processed/guvi_clean.json"



def load_data(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)



def clean_text(text):

    unwanted_phrases = [

        "login",
        "sign up",
        "signup",
        "apply now",
        "request a callback",
        "amazon vouchers",
        "iphone",
        "geekoins",
        "rewards",
        "leaderboard",
        "our team will reach",
        "within the next 24 hours",
        "explore now",
        "get started"

    ]


    for phrase in unwanted_phrases:

        text = re.sub(
            phrase,
            "",
            text,
            flags=re.IGNORECASE
        )


    # remove emojis

    text = re.sub(
        r'[^\w\s.,!?&()-]',
        '',
        text
    )


    # remove multiple spaces

    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()


def remove_bad_content(data):

    cleaned = []


    for item in data:

        content = item["content"]


        # ignore very small pages

        if len(content) < 300:
            continue


        # ignore pages with too much noise

        noise_words = [
            "login",
            "signup",
            "reward",
            "leaderboard"
        ]


        noise_count = sum(
            content.lower().count(word)
            for word in noise_words
        )


        if noise_count > 5:
            continue


        cleaned.append(item)


    return cleaned


def process():

    print(
        "Loading raw dataset..."
    )


    data = load_data(
        INPUT_FILE
    )


    print(
        "Raw samples:",
        len(data)
    )


    for item in data:

        item["content"] = clean_text(
            item["content"]
        )


    data = remove_bad_content(
        data
    )


    os.makedirs(
        "data/processed",
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


    print(
        "Clean samples:",
        len(data)
    )


    print(
        "Cleaning completed"
    )



if __name__ == "__main__":

    process()