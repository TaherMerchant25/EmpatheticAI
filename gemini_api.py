# gemini_api.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import sys

# --- Configuration ---
# Use the latest Flash model available
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
# GEMINI_MODEL_NAME = "models/gemini-1.5-flash-001" # Or be more specific

def load_gemini_llm(temperature=0.7):
    """
    Loads and initializes the Gemini LLM model.

    Args:
        temperature (float): Controls the randomness of the output.

    Returns:
        ChatGoogleGenerativeAI: An instance of the Gemini LLM.
        Returns None if the API key is missing.
    """
    load_dotenv()
    api_key = os.getenv("AIzaSyDIjEYblJJ242NAnkc2nfn_Mufo1RbSKYA")

    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please create a .env file with your key.")
        sys.exit(1) # Exit if key is absolutely required

    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            google_api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True # Often helpful for Gemini
            # You can add safety_settings here if needed
            # safety_settings=...
        )
        print(f"Successfully loaded Gemini model: {GEMINI_MODEL_NAME}")
        return llm
    except Exception as e:
        print(f"Error initializing Gemini LLM: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # Simple test to check if loading works
    print("Attempting to load Gemini LLM...")
    llm_instance = load_gemini_llm()
    if llm_instance:
        print("Gemini LLM loaded successfully for testing.")
    else:
        print("Failed to load Gemini LLM.")