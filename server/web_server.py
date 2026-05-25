from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from predictor import predict
import uvicorn
import numpy as np
import sys
import os
import threading
import time
import webbrowser
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ClassifierAPI")

app = FastAPI(
    title="ML Research Paper Classifier API",
    description="A multi-label classification system using scikit-learn models (Linear SVM) to categorize academic abstracts.",
    version="1.1.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema with validation
class TextPayload(BaseModel):
    text: str = Field(
        ..., 
        min_length=10, 
        max_length=15000, 
        description="The raw abstract text of the research paper to classify."
    )

@app.on_event("startup")
def startup_event():
    logger.info("Initializing ML models and checking NLTK downloads...")
    try:
        # Trigger an eager load and warm up prediction cache
        predict("Warm up query to initialize picked model and cached tokenizers.")
        logger.info("Model assets loaded successfully and API is ready to serve predictions.")
    except Exception as e:
        logger.error(f"Failed to initialize model assets on startup: {e}", exc_info=True)

@app.post("/api/classify")
def classify_text(payload: TextPayload):
    text = payload.text.strip()
    if not text:
        logger.warning("Received request with empty text payload.")
        raise HTTPException(status_code=400, detail="Text cannot be empty or contain only whitespace.")
    
    start_time = time.time()
    logger.info(f"Incoming prediction request (length: {len(text)} chars)")
    
    try:
        # Predict labels using the pipeline
        raw_predictions = predict(text)
        predictions = np.array(raw_predictions)[0]
        
        result = {
            "Computer Science": bool(predictions[0]),
            "Physics": bool(predictions[1]),
            "Mathematics": bool(predictions[2]),
            "Statistics": bool(predictions[3]),
            "Quantitative Biology": bool(predictions[4]),
            "Quantitative Finance": bool(predictions[5])
        }
        
        latency_ms = (time.time() - start_time) * 1000
        active_tags = [tag for tag, active in result.items() if active]
        logger.info(f"Prediction complete. Latency: {latency_ms:.2f}ms. Labels matched: {active_tags}")
        
        return {
            "predictions": result,
            "metadata": {
                "latency_ms": round(latency_ms, 2),
                "word_count": len(text.split()),
                "character_count": len(text)
            }
        }
    except Exception as e:
        logger.error(f"Error during classification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# Determine the absolute path to serve the built static Vite files
if getattr(sys, 'frozen', False):
    dist_dir = os.path.join(sys._MEIPASS, "dist")
else:
    dist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist")

if os.path.isdir(dist_dir):
    logger.info(f"Serving frontend static files from: {dist_dir}")
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_dir, "assets")), name="assets")
    
    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        # Serve index.html for all non-API paths ensuring SPA routing works
        path = os.path.join(dist_dir, full_path)
        if os.path.isfile(path) and not full_path.startswith("api/"):
            return FileResponse(path)
        return FileResponse(os.path.join(dist_dir, "index.html"))
else:
    logger.warning(f"Static directory not found at '{dist_dir}'. Local dev mode assumes React is run via npm dev server.")

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    if getattr(sys, 'frozen', False):
        threading.Thread(target=open_browser, daemon=True).start()
    
    logger.info("Starting FastAPI Uvicorn web server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
