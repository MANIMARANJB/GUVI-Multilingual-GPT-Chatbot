from models.language_detector import detect_language


texts = [
    "What is GUVI?",
    "GUVI பற்றி சொல்லுங்கள்",
    "GUVI के बारे में बताएं"
]


for text in texts:

    lang = detect_language(text)

    print("Text:", text)
    print("Language:", lang)
    print("----------------")
