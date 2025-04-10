import os
import uuid
from datetime import datetime
import streamlit as st
from database import SentimentVectorDatabase
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class MentalHealthChatbot:
    def __init__(self):
        """Initialize the chatbot application with sentiment database and Gemini integration."""
        # Initialize or retrieve user_id from session state
        if 'user_id' not in st.session_state:
            st.session_state.user_id = str(uuid.uuid4())
            st.session_state.start_time = datetime.now()
            st.session_state.sentiment_history = []

        # Initialize sentiment database
        self.sentiment_db = self.initialize_sentiment_db()

        # Initialize Gemini chatbot
        self.chatbot = self.initialize_chatbot()

    def initialize_sentiment_db(self):
        """Initialize and return the sentiment vector database."""
        db = SentimentVectorDatabase()
        if os.path.exists('sentiment_vectors.pkl'):
            try:
                db.load_database()
            except Exception as e:
                st.error(f"Could not load sentiment database. Reinitializing. Error: {str(e)}")
                db.save_database()
        else:
            db.save_database()
        return db

    def initialize_chatbot(self):
        """Initialize and return the Gemini chatbot instance for Gemini 2.0 Flash."""
        API_KEY = os.getenv("GEMINI_API_KEY",
                            "AIzaSyDPDDbuPWtUd5mskmQcxzGh4gjnGu8zV-0")  # Replace with your actual API key in .env file

        if not API_KEY:
            st.error("Gemini API key not found. Please provide a valid key.")
            return None

        genai.configure(api_key=API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        return gemini_model.start_chat(history=[])

    def analyze_sentiment(self, message):
        """Analyze the sentiment of a user message."""
        return self.sentiment_db.get_most_similar_sentiment(message, top_k=3)

    def process_user_message(self, message):
        """Process a user message, analyze sentiment, and generate response."""
        try:
            sentiment_data = self.analyze_sentiment(message) or []
            st.session_state.sentiment_history.append({
                "timestamp": datetime.now(),
                "message": message,
                "sentiments": sentiment_data
            })

            # Check for crisis indicators
            crisis_detected = any(pattern in message.lower() for pattern in [
                "suicide", "kill myself", "end my life", "harm myself",
                "hurt myself", "don't want to live"
            ])

            # Generate response using Gemini
            response = self.chatbot.send_message(message)
            response_text = response.text if hasattr(response, "text") else "I'm here to listen. Tell me more."

            # Add crisis resources
            if crisis_detected:
                response_text += (
                    "\n\n🚨 **IMPORTANT:** If you're in crisis, please reach out to these resources:\n"
                    "- 📞 National Suicide Prevention Lifeline: **988** or **1-800-273-8255**\n"
                    "- 📱 Crisis Text Line: **Text HOME to 741741**\n"
                    "- 🚑 Emergency: Call **911** (US) or your local emergency number"
                )

            return response_text
        except Exception as e:
            return f"An error occurred while processing your message: {str(e)}"


def main():
    st.set_page_config(
        page_title="EmpatheticAI - Mental Health Support",
        page_icon="❤️",
        layout="centered"
    )

    # Application title and description
    st.title("EmpatheticAI")
    st.markdown("A supportive AI companion for mental health")
    st.markdown("---")

    # Initialize the chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing EmpatheticAI..."):
            st.session_state.chatbot = MentalHealthChatbot()

        # Initialize chat history if it doesn't exist
        if 'messages' not in st.session_state:
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Hi there! I'm EmpatheticAI, a supportive companion designed to listen and provide mental health support. How are you feeling today?"
            }]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input area
    if user_input := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get and display bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.process_user_message(user_input)
                st.markdown(response)

        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar with info
    with st.sidebar:
        st.header("About EmpatheticAI")
        st.info("""
        EmpatheticAI is a supportive companion designed to provide mental health support.

        **Note:** This is not a replacement for professional mental health services.
        """)

        st.subheader("Crisis Resources")
        st.markdown("""
        - 📞 **National Suicide Prevention Lifeline:** 112 or 91-9820466726
        - 🚑 **Emergency:** Call Aasra: +91-9820466726 (IN) or your local emergency number
        """)

        if st.button("Clear Chat"):
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Hi there! I'm EmpatheticAI, a supportive companion designed to listen and provide mental health support. How are you feeling today?"
            }]
            st.rerun()


if __name__ == "__main__":
    main()