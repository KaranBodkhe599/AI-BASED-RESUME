import re
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded once
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation & numbers
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Tokenize and remove stopwords
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)
