import pandas as pd
import numpy as np
import pickle
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, hamming_loss
import nltk

# Ensure NLTK resources are available
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
from nltk.stem import WordNetLemmatizer
import contractions

stop_words = set(stopwords.words('english'))
lem = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Standardize to lowercase
    text = text.lower()
    # Expand contractions (e.g. don't -> do not)
    text = contractions.fix(text)
    # Tokenize words
    words = nltk.word_tokenize(text)
    # Lemmatize and remove stop words & punctuation
    cleaned = [
        lem.lemmatize(w) for w in words 
        if w.isalnum() and w not in stop_words
    ]
    return " ".join(cleaned)

def main():
    print("--- Starting Model Training Pipeline ---")
    
    # 1. Load original labeled dataset
    dataset_path = "train.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Please run the download script first.")
        return
    
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # 2. Text Preprocessing
    print("Cleaning abstracts (this might take a minute)...")
    start_time = time.time()
    df['Cleaned_Abstract'] = df['ABSTRACT'].apply(clean_text)
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds.")
    
    # Define multi-label targets
    label_cols = [
        'Computer Science', 'Physics', 'Mathematics', 
        'Statistics', 'Quantitative Biology', 'Quantitative Finance'
    ]
    X = df['Cleaned_Abstract']
    y = df[label_cols]
    
    # Save the preprocessed abstract features locally for sanity check/interviews
    df[['Cleaned_Abstract']].rename(columns={'Cleaned_Abstract': 'Abstract'}).to_csv('train_features.csv', index=True)
    print("Saved cleaned abstracts to train_features.csv")
    
    # 3. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Feature Extraction: TF-IDF with N-grams (1, 2)
    print("\nExtracting TF-IDF Features...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=15000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"TF-IDF shape: Train {X_train_vec.shape}, Test {X_test_vec.shape}")
    
    # 5. Dimensionality Reduction (Optional - TruncatedSVD)
    print("Running TruncatedSVD for experimentation (dim reduction)...")
    svd = TruncatedSVD(n_components=300, random_state=42)
    X_train_svd = svd.fit_transform(X_train_vec)
    X_test_svd = svd.transform(X_test_vec)
    
    # 6. Model Evaluation and Comparison
    results = []
    
    models = {
        "Logistic Regression (TF-IDF)": OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42)),
        "Linear SVM (TF-IDF)": OneVsRestClassifier(LinearSVC(random_state=42, dual=False)),
        "Logistic Regression (TF-IDF + SVD)": OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42)),
        "Linear SVM (TF-IDF + SVD)": OneVsRestClassifier(LinearSVC(random_state=42, dual=False))
    }
    
    # Train and evaluate each configuration
    for name, clf in models.items():
        print(f"Training: {name}...")
        start_train = time.time()
        
        # Use appropriate features
        if "SVD" in name:
            clf.fit(X_train_svd, y_train)
            preds = clf.predict(X_test_svd)
        else:
            clf.fit(X_train_vec, y_train)
            preds = clf.predict(X_test_vec)
            
        elapsed = time.time() - start_train
        
        # Calculate metrics
        h_loss = hamming_loss(y_test, preds)
        f1_micro = f1_score(y_test, preds, average='micro')
        f1_macro = f1_score(y_test, preds, average='macro')
        
        results.append({
            "Model": name,
            "F1 Micro": f1_micro,
            "F1 Macro": f1_macro,
            "Hamming Loss": h_loss,
            "Train Time (s)": elapsed
        })
        
    # Print results
    print("\n" + "="*70)
    print(f"{'Model Name':<35} | {'F1 Micro':<8} | {'F1 Macro':<8} | {'Hamming Loss':<12} | {'Time':<5}")
    print("="*70)
    for res in results:
        print(f"{res['Model']:<35} | {res['F1 Micro']:<8.4f} | {res['F1 Macro']:<8.4f} | {res['Hamming Loss']:<12.4f} | {res['Train Time (s)']:<5.2f}")
    print("="*70)
    
    # 7. Select and Save the Best Model
    # Looking at F1 Micro score, Linear SVM (TF-IDF) typically performs best on high-dimensional text.
    # We will fit the best model (Linear SVM without SVD) on the full dataset and save it.
    print("\nFitting the best model (Linear SVM (TF-IDF)) on the full dataset...")
    full_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=15000)
    X_full_vec = full_vectorizer.fit_transform(X)
    
    best_clf = OneVsRestClassifier(LinearSVC(random_state=42, dual=False))
    best_clf.fit(X_full_vec, y)
    
    # Save model and vectorizer
    model_file = "model.pkl"
    vectorizer_file = "vectorizer.pkl"
    
    with open(model_file, 'wb') as f:
        pickle.dump(best_clf, f)
    with open(vectorizer_file, 'wb') as f:
        pickle.dump(full_vectorizer, f)
        
    print(f"Saved best model to {model_file}")
    print(f"Saved fitted vectorizer to {vectorizer_file}")
    print("Training pipeline finished successfully!")

if __name__ == "__main__":
    main()
