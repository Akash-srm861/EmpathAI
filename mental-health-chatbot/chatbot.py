import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data (do this only once)
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Sample responses
responses = [
    "Hi there! How can I support you today?",
    "I'm here for you. Would you like to talk about how you're feeling?",
    "It’s okay to feel down sometimes. You are not alone.",
    "Would you like a breathing exercise or a motivational quote?",
    "Take your time. I'm listening.",
    "Everything will be okay. Stay strong 💪",
    "It’s important to talk to someone. Would you like help finding a resource?",
]

# Preprocess text
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    return " ".join(tokens)

# Get best matching response
def get_response(user_input):
    user_input = preprocess(user_input)
    all_text = responses + [user_input]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(all_text)
    similarity = cosine_similarity(tfidf[-1], tfidf[:-1])
    index = similarity.argmax()
    return responses[index]
