## Medical Chatbot
This project is an AI-powered Medical Chatbot designed to assist patients with instant, accurate, and multilingual medical support. It serves as a virtual healthcare assistant that can answer medical questions, analyze scanning reports, and give personalized health and wellness advice.

Built for patients, elderly individuals, and people with limited access to immediate medical consultation, this chatbot also supports visually impaired users through voice-based interaction. It can simplify complex health information and promote better health awareness, anytime and anywhere.

```markdown
🔍 Features

- 🧠 Medical Q&A Module: Answers medical-related questions using a trained LLM and using RAG Pipeline.
- 📄 Report Analysis Module: Reads medical scan reports (PDF/Image) and provides key insights finetuned LLM.
- 🌿 Health & Wellness Module: Suggests physical exercises, mental wellness tips, and health management advice.
- 🌐 Multi-language Support: Accepts and responds in multiple languages.
- 🎤 Voice Interface: Converts voice to text and text to voice using Speech-to-Text and Text-to-Speech.
- 💬 Minimal & Focused Responses: Gives concise answers, perfect for real-time assistance.

```

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/your-username/medical-chatbot.git
cd medical-chatbot
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the application**

```bash
streamlit run app.py
```
### Project Modules
```markdown

The medical chatbot consists of three main modules, each solving a specific problem in healthcare assistance:

---

'''
#### 1️⃣ Medical Q\&A Module

🔍 Purpose:
This module answers user queries related to health, symptoms, diseases, and general medical knowledge. It uses a Language Model (LLM) trained or prompted on medical data.

🛠️ How to Use:

* Type or speak your question in the chatbot (e.g., "What are the symptoms of diabetes?")
* The bot gives a short, accurate answer in your selected language.

👥 Helpful For:
Patients needing quick health info, people in remote areas, and those looking for trustworthy answers without searching online.

---

#### 2️⃣ Scanning Report Analysis Module

🔍 Purpose:
This module reads uploaded medical reports (PDF or image) and extracts important details. It uses OCR (Optical Character Recognition) to read text and NLP to interpret it.

🛠️ How to Use:

* Upload your scan report (e.g., blood test, CT scan, MRI Scan, X-Ray).
* The chatbot will read the text from the report and provide a simple explanation of key findings (e.g., "Your hemoglobin level is slightly low. It may indicate anemia.").

👥 Helpful For:
Patients who don’t understand medical terms in reports, elderly people, and those who want a second opinion in plain language.

---

#### 3️⃣ Holistic Health Management Module

🔍 Purpose:
This module gives daily health improvement suggestions like exercises, mental health tips, sleep routines, and food habits based on user needs or health conditions.

🛠️ How to Use:

* Ask for advice (e.g., "How can I reduce stress?" or "Suggest a workout plan for weight loss")
* The bot gives clear, customized suggestions.

👥 Helpful For:
Anyone looking to maintain good health, manage lifestyle diseases, or improve physical and mental well-being.

```


## 🛠️ Tech Stack

* **Frontend**: Streamlit
* **Language Models**: Google Gemini / Llama 3.3 70B / Llama 3.2 Vision 11B
* **Pipeline**: RAG
* **Speech**: gtts, pyttsx3
* **Translation**: Deeptranslator
* **Backend**: Python

## 👨‍⚕️ Use Cases

* Instant medical Q\&A in native languages
* Easy understanding of complex reports
* Daily health improvement tips
* Support for visually impaired users (voice-based interaction)



