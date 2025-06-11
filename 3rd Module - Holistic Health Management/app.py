import os
import streamlit as st
from dotenv import load_dotenv
import speech_recognition as sr  # Added for speech recognition

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Import the text-to-speech and translation functions
from text_to_speech_helper import text_to_speech
from translation import translate_text

load_dotenv()

GRQO_API_KEY = os.getenv("GROQ_API_KEY")
if not GRQO_API_KEY:
    raise ValueError("GROQ API Key is missing! Please add it to the .env file.")

# Initialize embedding model and database
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(
    collection_name="pharma_database",
    embedding_function=embedding_model,
    persist_directory='./pharma_db'
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

def run_rag_chain(query):
    retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 5})
    
    # Retrieve the current active chat's conversation history
    chat_history = "\n".join(st.session_state.chat_sessions[st.session_state.active_chat_id])
    
    PROMPT_TEMPLATE = f"""
    You are 🤖 HealthMate, an expert AI specializing in **holistic health management**.  
    Your goal is to provide **personalized suggestions** for:  
    - 🏋️ **Physical fitness** (exercise, posture, body pain relief)  
    - 🧘 **Mental wellness** (stress management, sleep improvement, mindfulness)  
    - 🥗 **Diet & nutrition** (healthy eating, hydration, meal planning)  
    - 🌿 **Lifestyle habits** (daily routines, habit formation, relaxation techniques)  

    ### 🔹 **Start the conversation by asking personalized questions:**  
    - "Hi! Before we begin, can you share a few details? 😊"  
    - "May I know your age range(25-30, 31-35 etc.,) and gender so I can give you the best recommendations?"  
    - "Do you have any specific health goals? (e.g., better sleep, weight management, reducing stress)"  
    - "How active is your daily routine? (Sedentary, Moderate, Highly Active)"  

    Based on the user's responses, tailor your advice to match their **specific needs**.  

    ✨ **Key Guidelines:**  
    - **DO NOT** answer queries about diseases, symptoms, or medical conditions.  
    - **Keep responses concise, friendly, and engaging**, using emojis to maintain a positive tone.  
    - **Offer practical, actionable advice** based on the user's inputs.  
    - **DO NOT provide medical treatments or diagnoses**—redirect the user to a doctor if necessary.  
    - **Ensure responses are motivating and supportive** to encourage healthy habits.  

    Chat History:  
    {chat_history}

    User 🧑: {{question}}
"""

    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chat_model = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GRQO_API_KEY,
        temperature=1
    )

    output_parser = StrOutputParser()

    # Use RunnableLambda to extract the "question" string and pass it to get_relevant_documents
    rag_chain = (
        {
            "context": RunnableLambda(lambda inp: retriever.get_relevant_documents(inp["question"])) | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | chat_model
        | output_parser
    )

    # Pass the query as a dict so that the chain receives {"question": query}
    response = rag_chain.invoke({"question": query})
    return response

def recognize_speech():
    """Capture speech from the microphone and return the recognized text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak now!")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio."
        except sr.RequestError as e:
            return f"Could not request results; {e}"

def init_session():
    """Initialize session state for multiple chats if not already set."""
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None
    if "chat_counter" not in st.session_state:
        st.session_state.chat_counter = 0
    # We'll use a separate key for voice input rather than modifying query_bottom directly.
    if "voice_input" not in st.session_state:
        st.session_state.voice_input = ""

def main():
    st.set_page_config(page_title="HealthMate", page_icon=":microscope:")

    init_session()

    # Inject custom CSS for spacing and styling
    st.markdown(
        """
        <style>
        audio {
            width: 300px !important;
            margin-top: 10px;
            margin-bottom: 5px;
        }
        .custom-title { 
            font-size: 46px; 
            text-align: center; 
            font-weight: bold; 
            font-family: Open Sans; 
            background: -webkit-linear-gradient(rgb(188, 12, 241), rgb(212, 4, 4)); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
        }
        .title-container {
            text-align: center;
        }
        .span { 
            font-size: 62px; 
        }
        .message-container {
            display: flex;
            margin: 10px 0;
            align-items: flex-start;
        }
        .message-container.bot {
            justify-content: flex-start;
        }
        .message-container.user {
            justify-content: flex-end;
        }
        .icon-container {
            font-size: 42px;
            line-height: 1;
            margin: 0 8px;
        }
        .user-message {
            background-color: #b5e550 !important;
            color: black !important;
            padding: 10px;
            border-radius: 10px;
            text-align: right;
            width: fit-content;
            max-width: 70%;
        }
        .bot-message {
            background-color: #b3cde0 !important;
            color: black !important;
            padding: 10px;
            border-radius: 10px;
            text-align: left;
            width: fit-content;
            max-width: 70%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title banner
    st.markdown(
        """
        <div class="title-container">
            <p class='span'><span class='custom-title'>HealthMate: Wellness & Lifestyle Coach</span></p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Sidebar Section
    with st.sidebar:
        if st.button("Open New Chat"):
            st.session_state.chat_counter += 1
            new_chat_id = f"Chat {st.session_state.chat_counter}s"
            st.session_state.chat_sessions[new_chat_id] = []
            st.session_state.active_chat_id = new_chat_id

        if st.session_state.chat_sessions:
            chat_ids = list(st.session_state.chat_sessions.keys())
            selected_chat = st.radio(
                "Previous Chat History", 
                chat_ids, 
                index=chat_ids.index(st.session_state.active_chat_id) if st.session_state.active_chat_id in chat_ids else 0
            )
            st.session_state.active_chat_id = selected_chat

        # Language selection
        languages = {
            "en": "English",
            "te": "తెలుగు",
            "ta": "தமிழ்",
            "kn": "ಕನ್ನಡ",
            "ml": "മലയാളം",
            "mr": "मराठी",
            "es": "Español",
            "fr": "Français",
            "de": "Deutsch",
            "hi": "हिन्दी",
            "zh": "中文"
        }
        language_names = list(languages.values())
        selected_language_name = st.selectbox("Select your language", language_names, index=0)
        user_lang = [code for code, name in languages.items() if name == selected_language_name][0]

        st.title("About HealthMate")
        st.info(
        "HealthMate is an AI-powered wellness assistant designed to provide **personalized guidance** on holistic health. "
        "It helps users with **fitness, mental wellness, nutrition, and lifestyle improvements**, offering tailored recommendations "
        "to support a healthier lifestyle. 🌿💪😊"
        )

        
        st.title("⚠️Disclaimer")
        st.warning(
            "Please note: The information provided here is for general informational purposes only and "
            "should not be taken as final advice. Always consult with a qualified healthcare provider for "
            "any recommendations related to medication or treatment."
        )
        
    if not st.session_state.active_chat_id:
        st.session_state.chat_counter += 1
        st.session_state.active_chat_id = f"Chat {st.session_state.chat_counter}"
        st.session_state.chat_sessions[st.session_state.active_chat_id] = []
    
    # Display conversation for the active chat session
    current_conversation = st.session_state.chat_sessions[st.session_state.active_chat_id]
    for i, chat_message in enumerate(current_conversation):
        if chat_message.startswith("🧑:"):
            user_text = chat_message.replace("🧑:", "").strip()
            st.markdown(
                f"""
                <div class="message-container user">
                    <div class="user-message">{user_text}</div>
                    <div class="icon-container">🤓</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif chat_message.startswith("🤖"):
            bot_text = chat_message.replace("🤖 HealthMate:", "").strip()
            st.markdown(
                f"""
                <div class="message-container bot">
                    <div class="icon-container">🤖</div>
                    <div class="bot-message">{bot_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Generate and play audio using the text-to-speech module
            audio_path = text_to_speech(bot_text)
            if audio_path:
                with open(audio_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
                st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)
                st.button(f"🔊 Listen", key=f"listen_{i}")
                st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)

    # Voice Input button
    if st.button("🎤 Voice Input"):
        spoken_text = recognize_speech()
        # Store recognized text in a separate key
        st.session_state.voice_input = spoken_text

    # Chat Form
    with st.form("chat_form", clear_on_submit=True):
        # Prepopulate with voice_input if available; otherwise, leave it empty.
        default_value = st.session_state.get("voice_input", "")
        query = st.text_input("Type your question here...", key="query_bottom", value=default_value)
        submitted = st.form_submit_button("Ask HealthMate")

        if submitted:
            if not query.strip():
                st.warning("Please enter a valid question.")
            else:
                # Translate query if necessary
                if user_lang != "en":
                    translated_query = translate_text(query, "en")
                else:
                    translated_query = query
                
                with st.spinner("Thinking..."):
                    english_response = run_rag_chain(query=translated_query)
                
                if user_lang != "en":
                    final_response = translate_text(english_response, user_lang)
                else:
                    final_response = english_response
                
                # Append messages to chat session
                st.session_state.chat_sessions[st.session_state.active_chat_id].append(f"🧑: {query}")
                st.session_state.chat_sessions[st.session_state.active_chat_id].append(f"🤖 HealthMate: {final_response}")
                
                # Remove the voice input so that text input starts empty next time.
                if "voice_input" in st.session_state:
                    del st.session_state["voice_input"]
                
                # No need to modify query_bottom here; clear_on_submit will reset it.

if __name__ == "__main__":
    main()
