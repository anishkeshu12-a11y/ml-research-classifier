# Academic Research Paper Classifier (Multi-Label NLP System)

A multi-label classification system that categorizes academic research papers into multiple scientific domains (Computer Science, Physics, Mathematics, Statistics, Quantitative Biology, and Quantitative Finance) based on their titles and abstracts.

The project features a **FastAPI backend** that serves model predictions, a **React (Vite) frontend** with a responsive dark-mode UI, and a **Python machine learning pipeline** using Scikit-Learn and NLTK.

---

## Directory Structure

```text
├── server/
│   ├── predictor.py         # NLP Preprocessing & model inference
│   ├── web_server.py        # FastAPI server & static UI mount
│   ├── requirements.txt     # Python dependencies
│   └── grpc_generated/      # gRPC service code (optional template folders)
├── web-ui/
│   ├── src/
│   │   ├── App.jsx          # React interface (API triggers & quick-test buttons)
│   │   ├── index.css        # Custom Glassmorphism styles
│   │   └── main.jsx
│   ├── package.json         # Frontend dependencies
│   └── vite.config.js       # Vite configuration
├── train.py                 # Pipeline evaluation, model comparison, & training script
├── train_features.csv       # Preprocessed abstract features (cached)
├── train.csv                # Raw labeled training dataset (ignored by Git)
├── model.pkl                # Trained OneVsRest Linear SVM model
├── vectorizer.pkl           # Trained TF-IDF Vectorizer
├── start_app.bat            # Quick launch script (Windows)
└── .gitignore               # Clean git exclusion rules
```

---

## Setup & Running the Project

### Prerequisite: Python 3.10+ & Node.js

### 1. Backend Setup & Model Training
1. Clone the repository and navigate to the project directory.
2. Install Python dependencies:
   ```bash
   pip install -r server/requirements.txt
   ```
3. *(Optional)* Retrain the model. The repo comes pre-packaged with `model.pkl` and `vectorizer.pkl`. If you wish to retrain or compare classifiers, run:
   ```bash
   python train.py
   ```
4. Start the FastAPI server:
   ```bash
   cd server
   python web_server.py
   ```
   The backend will start on `http://127.0.0.1:8000`.

### 2. Frontend Setup
1. Open a new terminal window and navigate to the `web-ui` directory:
   ```bash
   cd web-ui
   ```
2. Install frontend dependencies:
   ```bash
   npm install
   ```
3. Run the React development server:
   ```bash
   npm run dev
   ```
   This will start the frontend (typically on `http://localhost:5173`) and automatically open it in your browser.

---

## Model Comparison & Evaluation Metrics

The classification models were trained on the **Janatahack: Independence Day 2020 ML Hackathon** dataset (20,972 research papers). The text was preprocessed using lemmatization, contraction expansions, and stopword removal. 

Features were extracted using **TF-IDF with unigrams and bigrams** (max features capped at 15,000). Models were evaluated using an 80/20 train-test split:

| Model Configuration | F1-Score (Micro) | F1-Score (Macro) | Hamming Loss | Inference Speed |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (TF-IDF)** | 0.7979 | 0.6081 | 0.0793 | ~1.0 ms |
| **Linear SVM (TF-IDF)** | **0.7993** | **0.6745** | **0.0814** | **~1.2 ms** |
| **Logistic Regression (TF-IDF + SVD)** | 0.7911 | 0.5834 | 0.0820 | ~0.8 ms |
| **Linear SVM (TF-IDF + SVD)** | 0.8044 | 0.6580 | 0.0784 | ~2.5 ms |

### Key Findings:
- **Linear SVM (TF-IDF)** provides the best balance, achieving a strong **Macro F1-score of 0.6745**. It significantly outperforms Logistic Regression on low-frequency categories (like Quantitative Finance and Quantitative Biology).
- **Dimensionality Reduction (SVD)**: Using TruncatedSVD with 300 components condensed the feature space but resulted in a slight drop in Macro F1-score because it discarded fine-grained terminology patterns critical to niche domains.

---

## API Documentation

### Classify Document
*   **Endpoint:** `/api/classify`
*   **Method:** `POST`
*   **Request Headers:** `Content-Type: application/json`
*   **Request Body:**
    ```json
    {
      "text": "We formulate a new stochastic volatility model for pricing exotic options in incomplete markets. Using partial differential equations, we find semi-analytical solutions for European calls."
    }
    ```

*   **Response Body (200 OK):**
    ```json
    {
      "predictions": {
        "Computer Science": false,
        "Physics": false,
        "Mathematics": false,
        "Statistics": false,
        "Quantitative Biology": false,
        "Quantitative Finance": true
      },
      "metadata": {
        "latency_ms": 4.12,
        "word_count": 27,
        "character_count": 182
      }
    }
    ```

---

## Interview Q&A & Talking Points

Here are the key design decisions and optimizations to highlight in Data Science / ML interviews:

### 1. How did you optimize inference latency?
*   **The Bottleneck:** The original implementation re-loaded a 16MB raw features file and re-fitted the TF-IDF vectorizer on every single API request. This resulted in an unacceptable latency of 2 to 3 seconds per prediction.
*   **The Fix:** I refactored the pipeline to fit the vectorizer during the training phase and saved it as a serialized asset (`vectorizer.pkl`). I also implemented a global caching mechanism in FastAPI (`predictor.py`) to load the model and vectorizer into memory on startup. This reduced inference latency to **under 5 milliseconds** (a 500x speedup).

### 2. Why use TF-IDF + Linear SVM instead of deep learning (e.g. BERT)?
*   **Cost vs. Latency:** While a transformer model like BERT might offer a 2-3% improvement in F1-score, it requires a GPU for low-latency inference, leading to higher hosting costs.
*   **Practicality:** TF-IDF with bigrams captures scientific terminology very effectively, and a Linear SVM classifier runs in milliseconds on standard, inexpensive CPUs with a minimal RAM footprint. This makes it a highly practical, production-ready solution for standard enterprise workloads.

### 3. Why did you choose Linear SVM over Logistic Regression?
*   **High-Dimensional Margin:** TF-IDF text features are high-dimensional and sparse. Support Vector Machines search for the maximum-margin hyperplane, which is robust to overfitting in high-dimensional spaces. 
*   **Handling Class Imbalance:** Scientific labels are highly imbalanced (many Computer Science papers, very few Quantitative Finance papers). Linear SVM achieved a **Macro F1-score of 0.6745** compared to Logistic Regression's **0.6081**, indicating that the SVM is much better at identifying minority classes.
