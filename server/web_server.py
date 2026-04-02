from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from predictor import predict
import uvicorn
import numpy as np
import sys
import os
import threading
import time
import webbrowser

app = FastAPI(title="ML Project Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextPayload(BaseModel):
    text: str

@app.on_event("startup")
def startup_event():
    print("Testing ML system to pre-download any required NLTK corpuses...")
    try:
        predict("Startup test to initialize models and corpus downloads")
        print("ML Initialized")
    except Exception as e:
        print(f"Warning: ML Initialization test threw an exception: {e}")

@app.post("/api/classify")
def classify_text(payload: TextPayload):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    try:
        predictions = np.array(predict(text))[0]
        result = {
            "Computer Science": bool(predictions[0]),
            "Physics": bool(predictions[1]),
            "Mathematics": bool(predictions[2]),
            "Statistics": bool(predictions[3]),
            "Quantitative Biology": bool(predictions[4]),
            "Quantitative Finance": bool(predictions[5])
        }
        return {"predictions": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Determine the absolute path to the static 'dist' directory
if getattr(sys, 'frozen', False):
    # If running as PyInstaller executable, static files are unpacked here
    dist_dir = os.path.join(sys._MEIPASS, "dist")
else:
    # If running locally via python web_server.py
    dist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist")

if os.path.isdir(dist_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_dir, "assets")), name="assets")
    
    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        # Serve index.html for all non-API paths ensuring SPA routing works flawlessly
        path = os.path.join(dist_dir, full_path)
        if os.path.isfile(path) and not full_path.startswith("api/"):
            return FileResponse(path)
        return FileResponse(os.path.join(dist_dir, "index.html"))

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    # In Pyinstaller EXE environment, open the UI automatically
    if getattr(sys, 'frozen', False):
        threading.Thread(target=open_browser, daemon=True).start()
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
