from langdetect import detect


def detect_language(text):

    try:

        # English words check
        english_words = [
            "what",
            "who",
            "when",
            "where",
            "why",
            "how",
            "is",
            "are",
            "the",
            "guvi",
            "courses"
        ]

        words = text.lower().split()


        english_count = sum(
            1 for word in words
            if word in english_words
        )


        if english_count >= 2:
            return "en"


        return detect(text)


    except Exception:
        return "en"
