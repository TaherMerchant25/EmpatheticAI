import google.generativeai as genai
import os
import json
from datetime import datetime
import re


class GeminiChatbot:
    def __init__(self, api_key=""):
        """
        Initialize the Gemini chatbot.

        Args:
            api_key (str, optional): Google API key for Gemini. If None, looks for GOOGLE_API_KEY env variable.
        """
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if api_key is None:
                raise ValueError("API key must be provided or set as GOOGLE_API_KEY environment variable")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
        self.chat_history = {}
        self.safety_levels = {
            'harassment': 'block_medium_and_above',
            'hate_speech': 'block_medium_and_above',
            'sexually_explicit': 'block_only_high',
            'dangerous_content': 'block_medium_and_above'
        }
        self.setup_system_prompt()

    def setup_system_prompt(self):
        """Set up the system prompt for the therapeutic chatbot with guardrails."""
        self.system_prompt = """
        You are EmpatheticAI, a supportive AI chatbot designed to provide mental health support. Your purpose is to offer empathy, understanding, and guidance to users who are experiencing emotional distress or mental health challenges. Follow these principles:

        1. SUPPORT, NOT SUBSTITUTE: Always make it clear that you're a supportive tool, not a replacement for professional mental health care. Encourage seeking professional help when appropriate.

        2. EMPATHETIC ENGAGEMENT: Respond with genuine empathy and understanding. Recognize emotional cues and adapt your tone accordingly.

        3. PERSONALIZATION: Tailor your responses based on the user's needs, history, and preferences.

        4. INCLUSIVE SUPPORT: Be aware of and sensitive to diverse backgrounds, cultures, and experiences.

        5. CRISIS DETECTION: Identify potential crisis situations (suicidal ideation, self-harm, harm to others) and provide appropriate resources and escalation recommendations.

        6. ETHICAL BOUNDARIES: Never diagnose conditions, prescribe medications, or provide medical advice. Maintain user privacy and confidentiality.

        7. COMMUNICATION STYLE:
           - Use warm, conversational language
           - Validate feelings without judgment
           - Ask thoughtful questions to understand better
           - Provide reflective listening
           - Offer evidence-based coping strategies when appropriate
           - Be patient and allow users to express themselves fully

        8. RESOURCES: Suggest appropriate resources like crisis hotlines, mental health websites, or relaxation techniques when relevant.

        9. DOMAIN RESTRICTION: You are only allowed to discuss mental health topics. If a user asks something unrelated (e.g., technology, finance, entertainment), politely inform them that you can only provide mental health support.

        Remember: Your goal is to provide a safe, supportive space for users while encouraging professional help when needed.
        """

        self.chat = self.model.start_chat(history=[])
        self.chat.send_message(f"System: {self.system_prompt}")

    def filter_user_input(self, user_input):
        """Filter input to allow only mental health-related queries."""
        allowed_keywords = ["mental health", "stress", "anxiety", "depression", "coping", "therapy", "self-care",
                            "emotions", "feelings", "well-being"]

        if any(keyword in user_input.lower() for keyword in allowed_keywords):
            return self.chat.send_message(user_input)
        else:
            return "I'm here to support your mental well-being. If you have any concerns related to mental health, I'd be happy to help!"

    def generate_response(self, user_id, message, sentiment_data=None):
        """
        Generate a response for the user message.

        Args:
            user_id (str): Unique identifier for the user.
            message (str): User's message.
            sentiment_data (dict, optional): Sentiment analysis data to inform the response.

        Returns:
            dict: Response containing text and metadata.
        """
        # Initialize chat history for new users
        if user_id not in self.chat_history:
            self.chat_history[user_id] = []

        # Update context with sentiment if available
        sentiment_context = ""
        if sentiment_data:
            sentiment_context = f"\n[System note: User's message shows primary sentiment: {sentiment_data[0][0]} (confidence: {sentiment_data[0][1]:.2f}). Secondary sentiments: {', '.join([f'{s[0]}' for s in sentiment_data[1:]])}.]"

        # Add user's message to history
        self.chat_history[user_id].append({"role": "user", "content": message, "timestamp": datetime.now().isoformat()})

        # Create a temporary chat that includes the user's history and sentiment data
        temp_chat = self.model.start_chat(history=[])

        # Add system prompt
        temp_chat.send_message(f"System: {self.system_prompt}")

        # Brief history summary if there's enough context
        if len(self.chat_history[user_id]) > 3:
            recent_history = self.chat_history[user_id][-5:]  # Get last 5 messages
            history_summary = "Previous conversation summary:\n"
            for entry in recent_history:
                if entry["role"] == "user":
                    history_summary += f"User: {entry['content']}\n"
                else:
                    # Truncate assistant responses to keep summary brief
                    response_preview = entry["content"][:100] + "..." if len(entry["content"]) > 100 else entry[
                        "content"]
                    history_summary += f"You: {response_preview}\n"
            temp_chat.send_message(f"System: {history_summary}")

        # Send the user's message with sentiment context
        full_message = f"{message}{sentiment_context}"

        try:
            response = temp_chat.send_message(
                full_message,
                safety_settings=self.safety_levels,
                generation_config={"temperature": 0.7, "top_p": 0.95, "top_k": 40}
            )

            response_text = response.text if response else "I'm here to support you."

            # Analyze response for safety and user satisfaction metrics
            safety_metrics = self.analyze_response_safety(response_text)
            satisfaction_metrics = self.analyze_user_satisfaction(message, response_text)

            # Add AI's response to history
            self.chat_history[user_id].append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().isoformat(),
                "metrics": {**safety_metrics, **satisfaction_metrics}
            })

            return {
                "text": response_text,
                "metrics": {**safety_metrics, **satisfaction_metrics},
                "detected_issues": self.detect_critical_issues(message)
            }

        except Exception as e:
            error_msg = f"I'm sorry, I encountered an error processing your message. Please try again. Error: {str(e)}"

            self.chat_history[user_id].append({
                "role": "assistant",
                "content": error_msg,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })

            # Ensure 'metrics' key is present even in case of errors
            return {
                "text": error_msg,
                "metrics": {"error": True, "message": str(e)},
                "detected_issues": {}
            }

    def analyze_response_safety(self, response):
        """
        Analyze the safety aspects of the response.

        Args:
            response (str): The generated response text.

        Returns:
            dict: Safety metrics.
        """
        # Check for inappropriate advice indicators
        advisory_patterns = [
            r"you should definitely|you must|always|never|the only way",
            r"I recommend (?:that you|you|to) \w+ medications",
            r"stop (?:taking|using) (?:your|the) medication",
            r"diagnose[sd]? (?:with|as having) \w+"
        ]

        advisory_count = 0
        for pattern in advisory_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            advisory_count += len(matches)

        # Check for appropriate disclaimers
        has_disclaimer = bool(re.search(r"professional|therapist|doctor|healthcare", response, re.IGNORECASE))

        return {
            "advisory_count": advisory_count,
            "has_disclaimer": has_disclaimer,
            "overall_safety_score": max(0, 10 - advisory_count * 2 + (5 if has_disclaimer else 0))
        }

    def analyze_user_satisfaction(self, user_message, response):
        """
        Estimate potential user satisfaction with the response.

        Args:
            user_message (str): The user's message.
            response (str): The generated response.

        Returns:
            dict: Satisfaction metrics.
        """
        # Calculate response relevance (very simplified)
        user_words = set(user_message.lower().split())
        response_words = set(response.lower().split())
        relevant_words = user_words.intersection(response_words)
        relevance_score = len(relevant_words) / max(1, len(user_words)) * 10

        # Check for empathetic language
        empathy_patterns = [
            r"understand|hear you|sounds like|must be|feel",
            r"that's (?:tough|difficult|hard|challenging)",
            r"I'm (?:sorry|here for you)"
        ]

        empathy_count = 0
        for pattern in empathy_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            empathy_count += len(matches)

        # Check if response addresses questions
        question_in_user = bool(re.search(r"\?", user_message))
        if question_in_user:
            # Check if response likely answers a question
            likely_answers = bool(
                re.search(r"because|reason|consider|suggest|recommend|could be|might be", response, re.IGNORECASE))
        else:
            likely_answers = True  # No question to answer

        return {
            "relevance_score": min(10, relevance_score),
            "empathy_score": min(10, empathy_count * 2),
            "question_addressed": likely_answers,
            "estimated_satisfaction": (min(10, relevance_score) + min(10, empathy_count * 2) + (
                10 if likely_answers else 0)) / 3
        }

    def detect_critical_issues(self, message):
        """
        Detect potential critical issues in user messages.

        Args:
            message (str): The user's message.

        Returns:
            dict: Detected issues and their confidence.
        """
        issues = {}

        # Suicide/self-harm detection
        suicide_patterns = [
            r"(?:want|going|plan) to (?:die|kill|hurt|harm) (?:myself|me)",
            r"(?:end|take) my life",
            r"no (?:reason|point) (?:to live|in living)",
            r"better off (?:dead|without me)",
            r"suicide|suicidal"
        ]

        self_harm_patterns = [
            r"(?:cutting|harming|hurting) myself",
            r"(?:want|going) to hurt (?:myself|my body)",
            r"self[- ]harm",
            r"burn (?:myself|my skin)"
        ]

        crisis_patterns = [
            r"emergency|crisis|urgent|immediately|right now",
            r"(?:need|want) help (?:now|immediately|urgently)"
        ]

        violence_patterns = [
            r"(?:hurt|harm|kill) (?:someone|them|him|her|people)",
            r"(?:want|going) to (?:hurt|kill)",
            r"hurt other"
        ]

        # Check each pattern category
        for pattern in suicide_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                issues["suicide_risk"] = True
                break

        for pattern in self_harm_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                issues["self_harm_risk"] = True
                break

        for pattern in crisis_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                issues["crisis_situation"] = True
                break

        for pattern in violence_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                issues["violence_risk"] = True
                break

        return issues

    def save_chat_history(self, user_id, directory="chat_histories"):
        """
        Save a user's chat history to a file.

        Args:
            user_id (str): The user's ID.
            directory (str): Directory to save the history to.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = os.path.join(directory, f"{user_id}_{datetime.now().strftime('%Y%m%d')}.json")

        with open(filename, 'w') as f:
            json.dump(self.chat_history[user_id], f)

    def load_chat_history(self, user_id, filename):
        """
        Load a user's chat history from a file.

        Args:
            user_id (str): The user's ID.
            filename (str): The filename to load from.
        """
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.chat_history[user_id] = json.load(f)


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()  # Load API key from .env file

    # Initialize the chatbot
    chatbot = GeminiChatbot()

    # Test message
    user_id = "test_user_123"
    test_message = "I've been feeling really overwhelmed with work lately and I can't sleep at night."

    # Test sentiment data
    test_sentiment = [("anxiety", 0.85), ("overwhelmed", 0.75), ("fear", 0.45)]

    # Generate response
    response = chatbot.generate_response(user_id, test_message, test_sentiment)

    # Print response and metrics
    print("User:", test_message)
    print("Bot:", response["text"])
    print("\nMetrics:", response["metrics"])
    print("Detected issues:", response["detected_issues"])

    # Example of using filter_user_input method (moved from inside the class)
    test_filter_message = "Can you tell me about cryptocurrency?"
    filtered_response = chatbot.filter_user_input(test_filter_message)
    print("\nFilter test:", test_filter_message)
    print("Filter response:", filtered_response)
