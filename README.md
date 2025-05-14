# 🤖❤️ EmpatheticAI: Your Mental Health Companion

EmpatheticAI is an AI-powered, sentiment-aware mental health chatbot that offers therapeutic, non-judgmental conversations to individuals struggling with **depression**, **anxiety**, **trauma**, **stress**, and more. Built with Google's **Gemini AI**, **Streamlit**, and a custom **Sentiment Vector Database**, it provides real-time emotional support while integrating **RAG (Retrieval-Augmented Generation)** for enhanced contextual awareness.

> ⚠️ *This is not a replacement for professional therapy. Please seek professional help when in crisis.*

---

## 🌟 Features

- 🧠 **AI Therapy Chatbot** powered by **Gemini 1.5 Pro & Gemini Flash**
- 💬 **Streamlit UI** with conversational chat interface
- 📊 **Sentiment Analysis** using vector similarity and past emotional context
- 🔍 **RAG-Driven Context Awareness** to tailor responses to your emotional patterns
- 🚨 **Crisis Detection**: Identifies suicidal ideation and provides helpline resources
- 📚 **Conversation Memory**: Maintains chat history for a deeper, contextual connection
- 🧩 **Modular Design** for easy extensions (journaling, self-care tips, etc.)

---

## 🛠️ Tech Stack

- `Python`
- `Streamlit`
- `Google Generative AI (Gemini)`
- `scikit-learn` (for vector similarity)
- `dotenv` for secure API management
- `Pickle` for local database storage

---

## 🧪 Installation

First, ensure Python 3.8+ is installed.

### 1. Clone the repository

```yaml
git clone https://github.com/TaherMerchant25/EmpatheticAI.git
cd EmpatheticAI/Working_Code

```
### 2. Install Dependencies
```
pip install streamlit google-generativeai python-dotenv scikit-learn
```
### 3. Setup your environment variables
```
GEMINI_API_KEY=your-google-api-key-here
```
### 🚀 How to Run
```
streamlit run main_f.py
```
