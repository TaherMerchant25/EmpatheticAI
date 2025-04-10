import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from sentence_transformers import SentenceTransformer


class SentimentVectorDatabase:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the SentimentVectorDatabase.

        Args:
            model_name (str): The name of the Sentence Transformer model to use for embeddings.
        """
        self.encoder = SentenceTransformer(model_name)
        self.sentiment_vectors = {
            'joy': [],
            'sadness': [],
            'anger': [],
            'fear': [],
            'surprise': [],
            'disgust': [],
            'neutral': [],
            'anxiety': [],
            'hopelessness': [],
            'gratitude': [],
            'confusion': [],
            'overwhelmed': []
        }
        self.sample_phrases = {}
        self.initialize_vectors()

    def initialize_vectors(self):
        """Initialize the vector database with sample phrases for each sentiment category."""
        # Sample phrases for each sentiment
        self.sample_phrases = {
            'joy': [
                "I feel so happy today.",
                "Things are going really well.",
                "I'm excited about the future.",
                "That made me smile.",
                "I'm feeling great about my progress."
            ],
            'sadness': [
                "I feel really down today.",
                "I can't stop feeling sad.",
                "Everything feels hopeless.",
                "I'm struggling to find joy in anything.",
                "I feel empty inside."
            ],
            'anger': [
                "I'm so frustrated with this situation.",
                "This makes me really angry.",
                "I can't believe they would do that.",
                "I'm fed up with everything.",
                "I feel like I want to scream."
            ],
            'fear': [
                "I'm really anxious about this.",
                "I'm scared of what might happen.",
                "I can't stop worrying.",
                "I have this constant feeling of dread.",
                "I'm afraid things won't get better."
            ],
            'surprise': [
                "I didn't expect that at all.",
                "That completely caught me off guard.",
                "I'm shocked by what happened.",
                "I can't believe this is happening.",
                "This is such an unexpected turn of events."
            ],
            'disgust': [
                "That's really disturbing.",
                "I find that behavior repulsive.",
                "I'm disgusted by how I was treated.",
                "That makes me feel sick.",
                "I can't stand this situation."
            ],
            'neutral': [
                "I'm just taking it day by day.",
                "Things are neither good nor bad right now.",
                "I'm feeling okay, nothing special.",
                "I don't have strong feelings about it.",
                "I'm just observing what happens."
            ],
            'anxiety': [
                "My mind keeps racing with worries.",
                "I feel this constant knot in my stomach.",
                "I can't seem to relax at all.",
                "Everything feels overwhelming.",
                "I'm constantly on edge."
            ],
            'hopelessness': [
                "I don't see any way forward.",
                "Nothing I do seems to matter.",
                "I feel trapped in this situation.",
                "I've lost all hope for improvement.",
                "I can't imagine things ever getting better."
            ],
            'gratitude': [
                "I'm really thankful for the support I've received.",
                "I appreciate having someone to talk to.",
                "It means a lot that you're listening.",
                "I'm grateful for these small moments of peace.",
                "Thank you for being here for me."
            ],
            'confusion': [
                "I don't understand what's happening to me.",
                "My thoughts are all jumbled up.",
                "I can't make sense of my feelings.",
                "I'm lost and don't know what to do.",
                "Everything feels so confusing right now."
            ],
            'overwhelmed': [
                "There's just too much going on right now.",
                "I can't handle all of this.",
                "I feel buried under all these pressures.",
                "Everything is too much for me to deal with.",
                "I'm completely overwhelmed by my situation."
            ]
        }

        # Convert sample phrases to vectors and add some random variation
        np.random.seed(42)  # For reproducibility
        for sentiment, phrases in self.sample_phrases.items():
            base_embeddings = self.encoder.encode(phrases)

            # Create the base vectors
            for embedding in base_embeddings:
                self.sentiment_vectors[sentiment].append(embedding)

            # Add some variation to create more diverse vectors
            for _ in range(5):  # Add 5 variations for each original phrase
                for embedding in base_embeddings:
                    # Add small random variations (5% noise)
                    variation = embedding + np.random.normal(0, 0.05, embedding.shape)
                    # Normalize to unit length
                    variation = variation / np.linalg.norm(variation)
                    self.sentiment_vectors[sentiment].append(variation)

    def add_vector(self, text, sentiment):
        """
        Add a new vector to the database.

        Args:
            text (str): The text to encode and add to the database.
            sentiment (str): The sentiment category for the text.
        """
        if sentiment not in self.sentiment_vectors:
            raise ValueError(f"Sentiment '{sentiment}' not recognized.")

        embedding = self.encoder.encode(text)
        self.sentiment_vectors[sentiment].append(embedding)

    def get_most_similar_sentiment(self, text, top_k=3):
        """
        Find the most similar sentiment categories for a given text.

        Args:
            text (str): The text to analyze.
            top_k (int): Number of top sentiment categories to return.

        Returns:
            list: List of (sentiment, similarity_score) tuples.
        """
        query_embedding = self.encoder.encode(text)

        sentiment_scores = {}

        for sentiment, vectors in self.sentiment_vectors.items():
            if not vectors:
                continue

            # Calculate similarity with all vectors in this sentiment category
            similarities = cosine_similarity([query_embedding], vectors)[0]
            # Take the max similarity as the score for this sentiment
            sentiment_scores[sentiment] = np.max(similarities)

        # Sort by similarity score in descending order
        sorted_sentiments = sorted(sentiment_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_sentiments[:top_k]

    def get_random_vectors_for_sentiment(self, sentiment, count=3):
        """
        Return random vectors for a given sentiment.

        Args:
            sentiment (str): The sentiment category.
            count (int): Number of vectors to return.

        Returns:
            list: List of vectors.
        """
        if sentiment not in self.sentiment_vectors or not self.sentiment_vectors[sentiment]:
            return []

        vectors = self.sentiment_vectors[sentiment]
        indices = np.random.choice(len(vectors), min(count, len(vectors)), replace=False)
        return [vectors[i] for i in indices]

    def save_database(self, filepath='sentiment_vectors.pkl'):
        """
        Save the database to a file.

        Args:
            filepath (str): Path to save the database.
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'sentiment_vectors': self.sentiment_vectors,
                'sample_phrases': self.sample_phrases
            }, f)
        print(f"Database saved to {filepath}")

    def load_database(self, filepath='sentiment_vectors.pkl'):
        """
        Load the database from a file.

        Args:
            filepath (str): Path to load the database from.
        """
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.sentiment_vectors = data['sentiment_vectors']
                self.sample_phrases = data['sample_phrases']
            print(f"Database loaded from {filepath}")
        else:
            print(f"File {filepath} not found. Using default database.")

    def get_sentiment_suggestions(self, sentiment, count=3):
        """
        Get sample phrases for a given sentiment.

        Args:
            sentiment (str): The sentiment category.
            count (int): Number of phrases to return.

        Returns:
            list: List of sample phrases.
        """
        if sentiment not in self.sample_phrases:
            return []

        phrases = self.sample_phrases[sentiment]
        return np.random.choice(phrases, min(count, len(phrases)), replace=False).tolist()


# Example usage
if __name__ == "__main__":
    db = SentimentVectorDatabase()

    # Test the database
    test_texts = [
        "I'm feeling really anxious about my future",
        "Today was a wonderful day, everything went well",
        "I'm so angry about how I was treated",
        "I don't know what to do anymore, nothing seems to work"
    ]

    for text in test_texts:
        top_sentiments = db.get_most_similar_sentiment(text)
        print(f"Text: {text}")
        print(f"Top sentiments: {top_sentiments}")

        # Get suggestions for the top sentiment
        top_sentiment = top_sentiments[0][0]
        suggestions = db.get_sentiment_suggestions(top_sentiment)
        print(f"Suggestions for {top_sentiment}: {suggestions}")
        print("---")

    # Save the database
    db.save_database()