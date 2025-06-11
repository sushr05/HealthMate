from deep_translator import GoogleTranslator

def translate_text(text, target_language="en"):
    """
    Translates the given text into the specified target language.
    
    Args:
        text (str): The text to be translated.
        target_language (str): The target language code (default is "en").

    Returns:
        str: The translated text.
    """
    if target_language == "en":  # No translation needed for English
        return text
    
    try:
        translated_text = GoogleTranslator(source="en", target=target_language).translate(text)
        return translated_text
    except Exception as e:
        return f"Translation Error: {e}"
