import os
import re
from io import BytesIO
from PIL import Image
import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from scholarly import scholarly
from deep_translator import GoogleTranslator
from gtts import gTTS
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Page configuration
st.set_page_config(page_title="HealthMate", page_icon=":microscope:", layout="wide")

# Initialize session state variables if not already set
if "target_language" not in st.session_state:
    st.session_state["target_language"] = "en"
if "tts_enabled" not in st.session_state:
    st.session_state["tts_enabled"] = True

# Sidebar configuration
with st.sidebar:
    st.title("ℹ Configuration")

    if not GOOGLE_API_KEY:
        st.error("Missing API Key. Please add it to your .env file.")
    else:
        st.success("API Key loaded successfully from .env")

    st.session_state["target_language"] = st.selectbox(
        "Select Output Language",
        options=["en", "es", "fr", "de", "it", "pt", "te", "ta", "kn", "ml", "mr", "hi"],
        format_func=lambda x: {
            "en": "English", "es": "Spanish", "fr": "French", "de": "German",
            "it": "Italian", "pt": "Portuguese", "te": "తెలుగు", "ta": "தமிழ்",
            "kn": "ಕನ್ನಡ", "ml": "മലയാളം", "mr": "मराठी", "hi": "हिन्दी"
        }.get(x, x),
    )

    st.session_state["tts_enabled"] = st.checkbox("Enable Text-to-Speech", value=True)

    st.title("About HealthMate")
    st.info(
    "🩺 *Welcome to HealthMate – Your AI-Powered Medical Imaging Assistant!*\n\n"
    "HealthMate is an advanced AI-driven tool designed to assist patients and healthcare professionals in analyzing medical images such as X-rays, MRIs, CT scans, and ultrasounds. "
    "By leveraging cutting-edge AI models and real-time research integration, HealthMate provides detailed medical insights while ensuring accessibility through multilingual support and text-to-speech functionality.\n\n"
    
    "### 🌟 *Key Features:*\n"
    "✅ *AI-Powered Image Analysis* – Detects abnormalities, identifies affected regions, and provides a structured diagnosis.\n\n"
    "✅ *Research-Backed Insights* – Fetches the latest medical references from Google Scholar for informed decision-making.\n\n"
    "✅ *Multilingual Support* – Delivers explanations in multiple languages to enhance accessibility for diverse users.\n\n"
    "✅ *Patient-Friendly Explanations* – Simplifies complex medical terminology for easy understanding.\n\n"
    "✅ *Text-to-Speech (TTS) Output* – Converts AI-generated reports into speech for enhanced usability.\n\n"

    "### 🔍 *How HealthMate Helps You:*\n"
    "🔹 *Patients & Caregivers* – Understand medical imaging results with clear, non-technical explanations.\n\n"
    "🔹 *Doctors & Radiologists* – Get AI-assisted second opinions and quick research-based references.\n\n"
    "🔹 *Medical Students & Researchers* – Access AI-driven diagnostic insights alongside the latest medical literature.\n\n"

)



# Initialize Medical AI Agent
medical_agent = None
if GOOGLE_API_KEY:
    medical_agent = Agent(model=Gemini(api_key=GOOGLE_API_KEY, id="gemini-2.0-flash-exp"), markdown=True)
else:
    st.warning("Please configure your API key in the .env file to continue.")

# Function to search Google Scholar
def search_google_scholar(query, max_results=3):
    try:
        search_query = scholarly.search_pubs(query)
        results = []
        for _ in range(max_results):
            try:
                pub = next(search_query)
                results.append({
                    "title": pub.get("bib", {}).get("title", "No title"),
                    "year": pub.get("bib", {}).get("pub_year", "Unknown Year"),
                    "url": pub.get("pub_url", "No URL"),
                })
            except StopIteration:
                break
        return results
    except Exception as e:
        st.error(f"Error fetching Google Scholar results: {e}")
        return []

# Medical Analysis Query
query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the patient's medical image and structure your response as follows:

### 1. Image Type & Region
- Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
- Identify the patient's anatomical region and positioning
- Comment on image quality and technical adequacy

### 2. Key Findings
- List primary observations systematically
- Note any abnormalities in the patient's imaging with precise descriptions
- Include measurements and densities where relevant
- Describe location, size, shape, and characteristics
- Rate severity: Normal/Mild/Moderate/Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses in order of likelihood
- Support each diagnosis with observed evidence from the patient's imaging
- Note any critical or urgent findings

### 4. Patient-Friendly Explanation
- Explain the findings in simple, clear language that the patient can understand
- Avoid medical jargon or provide clear definitions
- Include visual analogies if helpful
- Address common patient concerns related to these findings

### 5. Research Context
- Find medical literature about similar cases
- Search for standard treatment protocols
- Provide a list of relevant medical links of them too
- Research any relevant technological advances
- Include 2-3 key references to support your analysis

Format your response using clear markdown headers and bullet points. Be concise yet thorough.
"""


# UI Header Styling
st.markdown("""
    <style>
    .custom-title { 
        font-size: 46px; 
        text-align: center; 
        font-weight: bold;
        font-family: Open Sans;
        background: -webkit-linear-gradient(rgb(188, 12, 241), rgb(212, 4, 4));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .title-container { text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-container">
        <p><span class='custom-title'>HealthMate: Scan Report Analyzer</span></p>
    </div>
""", unsafe_allow_html=True)

st.write("Upload a medical image for professional analysis.")

# Image Upload
uploaded_file = st.file_uploader("Upload Medical Image", type=["jpg", "jpeg", "png", "dicom"])

if uploaded_file:
    # Display Image
    image = Image.open(uploaded_file)
    width, height = image.size
    aspect_ratio = width / height
    resized_image = image.resize((450, int(450 / aspect_ratio)))  # Adjust size as needed
    st.image(resized_image, caption="Uploaded Medical Image", use_column_width=False)

    analyze_button = st.button("🔍 Analyze Image", type="primary")

    if analyze_button:
        image_path = "temp_medical_image.png"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("🔄 Analyzing image... Please wait."):
            try:
                # Run AI analysis
                response = medical_agent.run(query, images=[image_path])
                
                # Add research context
                scholar_results = search_google_scholar("radiology diagnostic imaging treatment protocols")
                research_md = "\n\n### 5. Research Context\n\nRecent research and treatment guidelines:\n"
                for res in scholar_results:
                    research_md += f"- [{res['title']}]({res['url']}) ({res['year']})\n"

                final_response = response.content + research_md

                # Translate output if needed
                if st.session_state["target_language"] != "en":
                    try:
                        translated_text = GoogleTranslator(
                            source="en", target=st.session_state["target_language"]
                        ).translate(final_response)
                        final_response = translated_text if translated_text else final_response
                    except Exception as e:
                        st.error(f"Translation error: {e}")

                # Display analysis
                st.markdown("### 📋 Analysis Results")
                st.markdown(final_response)
                st.caption("Note: AI-generated analysis should be reviewed by a healthcare professional.")

                # Text-to-Speech if enabled
                if st.session_state["tts_enabled"]:
                    try:
                        tts_text = re.sub(r"[\*\[\]\(\)#@,]", "", final_response)
                        tts = gTTS(text=tts_text, lang=st.session_state["target_language"])
                        tts_fp = BytesIO()
                        tts.write_to_fp(tts_fp)
                        tts_fp.seek(0)
                        st.audio(tts_fp, format="audio/mp3")
                    except Exception as e:
                        st.error(f"Text-to-Speech error: {e}")

            except Exception as e:
                st.error(f"Analysis error: {e}")

            finally:
                os.remove(image_path)
else:
    st.info("👆 Please upload a medical image to begin analysis.")