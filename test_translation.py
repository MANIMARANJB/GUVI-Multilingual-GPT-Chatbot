from models.translator import (
    translate_to_english,
    translate_from_english
)


text = "GUVI பற்றி சொல்லுங்கள்"


english = translate_to_english(
    text,
    "ta"
)


print("English Translation:")
print(english)


tamil = translate_from_english(
    "GUVI is a technology learning platform.",
    "ta"
)


print("\nTamil Translation:")
print(tamil)
