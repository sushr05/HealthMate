import tempfile
import re
from gtts import gTTS
import langdetect  

def clean_text(text):
    """
    Remove unnecessary special characters but keep non-English scripts.
    """
    text = re.sub(r'[^\w\s.,!?₹€À-ÖØ-öø-ÿ\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F]', '', text)  
    return text

def detect_language(text):
    """Detect language dynamically."""
    try:
        return langdetect.detect(text)
    except:
        return "en"  # Default to English if detection fails

def text_to_speech(response_text):
    """
    Convert cleaned text to speech and return the audio file path.
    """
    try:
        cleaned_text = clean_text(response_text)  
        detected_lang = detect_language(cleaned_text)  

        # Handle regional language codes
        language_map = {
            "zh": "zh-cn",  # Convert generic Chinese to Mandarin
            "pt": "pt-br",  # Convert Portuguese to Brazilian Portuguese
            "te": "te",  # Ensure Telugu is correctly recognized
        }
        lang_code = language_map.get(detected_lang, detected_lang)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio_path = temp_audio.name
            tts = gTTS(text=cleaned_text, lang=lang_code, slow=True)  # Slow=True for better pronunciation
            tts.save(temp_audio_path)
        
        return temp_audio_path
    except Exception as e:
        print(f"Error in Text-to-Speech: {e}")
        return None
