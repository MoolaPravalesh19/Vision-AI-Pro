# backend/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = FastAPI()

# Enable CORS so your React frontend can talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the local YOLOv11L model
# This will download the weights automatically on the first run
model = YOLO("yolo11m.pt") 

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # Read the image from the frontend
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run YOLOv11L inference
    results = model(img)[0]
    
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "label": results.names[int(box.cls[0])],
            "score": float(box.conf[0]),
            "classId": int(box.cls[0])
        })
    
    return {"detections": detections}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)