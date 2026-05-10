// Dynamically imported on the client to avoid SSR `document`/`window` access.
type Ort = typeof import("onnxruntime-web");
let ort: Ort | null = null;

async function getOrt(): Promise<Ort> {
  if (ort) return ort;
  const mod = await import("onnxruntime-web");
  mod.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/";
  mod.env.wasm.numThreads = 1;
  ort = mod;
  return mod;
}

const MODEL_URL =
  "https://cdn.jsdelivr.net/gh/Hyuto/yolov8-onnxruntime-web@master/public/model/yolov8n.onnx";

export const INPUT_SIZE = 640;

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
  // box in source-image pixel coords (xyxy)
  x1: number; y1: number; x2: number; y2: number;
};

type InferenceSession = import("onnxruntime-web").InferenceSession;
let sessionPromise: Promise<InferenceSession> | null = null;

export function loadModel() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const o = await getOrt();
      return o.InferenceSession.create(MODEL_URL, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
    })();
  }
  return sessionPromise;
}

// Letterbox + normalize source frame into Float32 NCHW tensor data
function preprocess(
  source: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement,
  srcW: number,
  srcH: number,
  ctx: CanvasRenderingContext2D,
) {
  const r = Math.min(INPUT_SIZE / srcW, INPUT_SIZE / srcH);
  const newW = Math.round(srcW * r);
  const newH = Math.round(srcH * r);
  const padX = Math.floor((INPUT_SIZE - newW) / 2);
  const padY = Math.floor((INPUT_SIZE - newH) / 2);

  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.drawImage(source, padX, padY, newW, newH);
  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);

  const N = INPUT_SIZE * INPUT_SIZE;
  const out = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    out[i] = data[i * 4] / 255;             // R
    out[i + N] = data[i * 4 + 1] / 255;     // G
    out[i + 2 * N] = data[i * 4 + 2] / 255; // B
  }
  return { tensorData: out, r, padX, padY };
}

function iou(a: Detection, b: Detection) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-6);
}

function nms(dets: Detection[], iouThresh = 0.45) {
  const sorted = [...dets].sort((a, b) => b.score - a.score);
  const keep: Detection[] = [];
  while (sorted.length) {
    const best = sorted.shift()!;
    keep.push(best);
    for (let i = sorted.length - 1; i >= 0; i--) {
      if (sorted[i].classId === best.classId && iou(best, sorted[i]) > iouThresh) {
        sorted.splice(i, 1);
      }
    }
  }
  return keep;
}

export async function detect(
  source: HTMLVideoElement,
  workCtx: CanvasRenderingContext2D,
  scoreThresh = 0.35,
): Promise<Detection[]> {
  const session = await loadModel();
  const o = await getOrt();
  const srcW = source.videoWidth;
  const srcH = source.videoHeight;
  if (!srcW || !srcH) return [];

  const { tensorData, r, padX, padY } = preprocess(source, srcW, srcH, workCtx);
  const input = new o.Tensor("float32", tensorData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const results = await session.run({ [inputName]: input });
  const output = results[outputName];
  const data = output.data as Float32Array;
  // YOLOv8 output: [1, 84, 8400] => 4 box + 80 classes
  const dims = output.dims; // e.g. [1, 84, 8400]
  const numClasses = dims[1] - 4;
  const numBoxes = dims[2];

  const dets: Detection[] = [];
  for (let i = 0; i < numBoxes; i++) {
    let bestC = -1;
    let bestS = 0;
    for (let c = 0; c < numClasses; c++) {
      const s = data[(4 + c) * numBoxes + i];
      if (s > bestS) {
        bestS = s;
        bestC = c;
      }
    }
    if (bestS < scoreThresh) continue;

    const cx = data[0 * numBoxes + i];
    const cy = data[1 * numBoxes + i];
    const w = data[2 * numBoxes + i];
    const h = data[3 * numBoxes + i];
    // map back from letterboxed 640 to source coords
    const x1 = (cx - w / 2 - padX) / r;
    const y1 = (cy - h / 2 - padY) / r;
    const x2 = (cx + w / 2 - padX) / r;
    const y2 = (cy + h / 2 - padY) / r;

    dets.push({
      classId: bestC,
      label: COCO_LABELS[bestC] ?? `cls_${bestC}`,
      score: bestS,
      x1: Math.max(0, x1),
      y1: Math.max(0, y1),
      x2: Math.min(srcW, x2),
      y2: Math.min(srcH, y2),
    });
  }
  return nms(dets, 0.45);
}
