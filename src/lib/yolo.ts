/**
 * src/lib/yolo.ts 
 * Updated to use Local Python Backend (YOLOv11L)
 */

export const COCO_LABELS = [
  "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
  "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
  "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
  "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
  "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
  "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
  "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
  "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
  "remote","keyboard","cell phone","microwave","oven","toaster","sink",
  "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
  "toothbrush",
];

export type Detection = {
  label: string;
  classId: number;
  score: number;
  x1: number; y1: number; x2: number; y2: number;
};

// We keep this for UI compatibility, but the backend handles the actual scaling
export const INPUT_SIZE = 640; 

/**
 * In the new setup, the backend manages the model.
 * We use this function to check if the local server is alive.
 */
export async function loadModel() {
  try {
    const res = await fetch("http://localhost:8000/health"); // Optional health check endpoint
    if (!res.ok) throw new Error("Backend not responding");
    return true;
  } catch (err) {
    console.warn("Backend not found, ensure your Python server is running on port 8000");
    return Promise.resolve(true); // Return resolved to not block the UI
  }
}

/**
 * Sends a video frame to the local Python FastAPI backend.
 * The backend performs YOLOv11L inference and returns results.
 */
export async function detect(
  source: HTMLVideoElement,
  _workCtx?: CanvasRenderingContext2D, // No longer strictly needed for preprocessing
  _scoreThresh = 0.35,
): Promise<Detection[]> {
  const srcW = source.videoWidth;
  const srcH = source.videoHeight;
  
  if (!srcW || !srcH) return [];

  // 1. Capture the current frame from the video to a Canvas
  const canvas = document.createElement("canvas");
  canvas.width = srcW;
  canvas.height = srcH;
  const ctx = canvas.getContext("2d");
  if (!ctx) return [];
  ctx.drawImage(source, 0, 0, srcW, srcH);

  // 2. Convert Canvas to Blob (JPEG) to send over HTTP
  return new Promise((resolve) => {
    canvas.toBlob(async (blob) => {
      if (!blob) return resolve([]);

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      try {
        // 3. POST request to your FastAPI server
        const response = await fetch("http://localhost:8000/detect", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) throw new Error("Inference failed");

        const data = await response.json();
        
        // Data format from backend: { detections: Detection[] }
        resolve(data.detections);
      } catch (err) {
        console.error("Local Backend Error:", err);
        resolve([]);
      }
    }, "image/jpeg", 0.8); // 0.8 quality provides a good balance of speed and accuracy
  });
}