from langdetect import detect, DetectorFactory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# LANGUAGE DETECTION CONFIGURATION

DetectorFactory.seed = 0


# TRANSLATION MODEL

TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"


# LANGUAGE MAPPING

LANGUAGE_MAP = {
    "en": "eng_Latn",   # English
    "ta": "tam_Taml",   # Tamil
    "hi": "hin_Deva",   # Hindi
    "te": "tel_Telu",   # Telugu
    "kn": "kan_Knda",   # Kannada
    "ml": "mal_Mlym"    # Malayalam
}


# LOAD MODEL

tokenizer = AutoTokenizer.from_pretrained(
    TRANSLATION_MODEL_NAME
)

translation_model = AutoModelForSeq2SeqLM.from_pretrained(
    TRANSLATION_MODEL_NAME
)


# LANGUAGE DETECTION

def detect_language(text):
    """Detect the language of the given text."""

    try:
        return detect(text)
    except Exception:
        return "en"


# TRANSLATION

def translate_text(text, source_lang, target_lang):
    """Translate text from source language to target language."""

    if source_lang not in LANGUAGE_MAP:
        return text

    if target_lang not in LANGUAGE_MAP:
        return text

    if source_lang == target_lang:
        return text

    source_code = LANGUAGE_MAP[source_lang]
    target_code = LANGUAGE_MAP[target_lang]

    tokenizer.src_lang = source_code

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    translated_tokens = translation_model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(
            target_code
        ),
        max_length=512
    )

    translated_text = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]

    return translated_text


# TRANSLATE TO ENGLISH

def translate_to_english(text):
    """Detect language and translate user input into English."""

    detected_lang = detect_language(text)

    if detected_lang == "en":
        return text, detected_lang

    if detected_lang not in LANGUAGE_MAP:
        return text, detected_lang

    english_text = translate_text(
        text=text,
        source_lang=detected_lang,
        target_lang="en"
    )

    return english_text, detected_lang


# TRANSLATE FROM ENGLISH

def translate_from_english(text, target_lang):
    """Translate the English response back to the user's language."""

    if target_lang == "en":
        return text

    if target_lang not in LANGUAGE_MAP:
        return text

    translated_text = translate_text(
        text=text,
        source_lang="en",
        target_lang=target_lang
    )

    return translated_text