from deep_translator import GoogleTranslator



def translate_to_english(text, source_lang):

    if source_lang == "en":
        return text

    translated = GoogleTranslator(
        source=source_lang,
        target="en"
    ).translate(text)

    return translated



def translate_from_english(text, target_lang):

    if target_lang == "en":
        return text

    translated = GoogleTranslator(
        source="en",
        target=target_lang
    ).translate(text)

    return translated
