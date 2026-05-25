import pickle
import pandas as pd
import numpy as np
import nltk
import os

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

from nltk.corpus import stopwords
import contractions
from nltk import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lem = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Expand contractions
    text = contractions.fix(text)
    # Word tokenization
    words = nltk.word_tokenize(text)
    # Remove stopwords, punctuation, and keep alphanumeric tokens
    cleaned = [
        lem.lemmatize(w) for w in words 
        if w.isalnum() and w not in stop_words
    ]
    return " ".join(cleaned)

def process_text(df):
    """
    Applies text cleaning to the 'Abstract' column of the input dataframe.
    """
    cleaned_abstracts = df['Abstract'].apply(clean_text)
    return pd.DataFrame(cleaned_abstracts, columns=['Abstract'])

# Global cache to avoid reloading models from disk on every API call
model_cache = None
vectorizer_cache = None

def load_resources():
    """
    Loads the trained model and vectorizer relative to the script location.
    Caches them globally to enable sub-millisecond inference.
    """
    global model_cache, vectorizer_cache
    if model_cache is None or vectorizer_cache is None:
        # Predictor is in 'server' or 'mlproject' folder. Root is one level up.
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'model.pkl')
        vectorizer_path = os.path.join(base_dir, 'vectorizer.pkl')
        
        # Fallback to local paths if run independently
        if not os.path.exists(model_path):
            model_path = 'model.pkl'
        if not os.path.exists(vectorizer_path):
            vectorizer_path = 'vectorizer.pkl'
            
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Trained model files not found. Please train the model first by running 'python train.py'. "
                f"Expected paths: {model_path} and {vectorizer_path}"
            )
            
        with open(model_path, 'rb') as f:
            model_cache = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer_cache = pickle.load(f)

def predict(text: str):
    """
    Predicts the multi-label topics for a given input research abstract.
    """
    load_resources()
    
    # Preprocess text using the simplified pipeline
    test_df = pd.DataFrame([text], columns=['Abstract'])
    test_df = process_text(test_df)
    
    # Transform abstract using pre-fitted vectorizer
    vectors = vectorizer_cache.transform(test_df['Abstract'])
    
    # Predict topics
    preds = model_cache.predict(vectors)
    
    # Convert sparse predictions to matrix format to match downstream callers
    if hasattr(preds, 'todense'):
        return preds.todense()
    else:
        return np.matrix(preds)