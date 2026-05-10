import { useRef, useState, useEffect, useCallback } from "react";
import Webcam from "react-webcam";
import {
  FiCpu, FiActivity, FiVideo, FiVideoOff, FiZap, FiEye, FiCamera, FiWifi,
} from "react-icons/fi";
import { detect, loadModel, INPUT_SIZE, type Detection } from "./lib/yolo";

const BOX_COLORS = [
  "#22d3ee","#a78bfa","#f472b6","#facc15","#34d399",
  "#fb7185","#60a5fa","#f97316","#c084fc","#4ade80",
];

function Stat({
  label, value, icon: Icon, accent,
}: {
  label: string;
  value: string;
  icon: React.ComponentType<{ className?: string }>;
  accent?: boolean;
}) {
  return (
    <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4 flex items-center gap-3">
      <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
        accent ? "bg-cyan-500 text-slate-950" : "bg-slate-800 text-cyan-400"
      }`}>
        <Icon className="text-lg" />
      </div>
      <div className="min-w-0">
        <p className="text-[10px] uppercase tracking-widest text-slate-400">{label}</p>
        <p className="text-sm font-semibold text-white truncate">{value}</p>
      </div>
    </div>
  );
}

export default function App() {
  const webcamRef = useRef<Webcam>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const workCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const runningRef = useRef(false);

  const [isLive, setIsLive] = useState(false);
  const [fps, setFps] = useState(0);
  const [latency, setLatency] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [modelStatus, setModelStatus] =
    useState<"idle" | "loading" | "ready" | "error">("idle");
  const [modelError, setModelError] = useState<string | null>(null);

  const getWorkCtx = useCallback(() => {
    if (!workCanvasRef.current) {
      const c = document.createElement("canvas");
      c.width = INPUT_SIZE;
      c.height = INPUT_SIZE;
      workCanvasRef.current = c;
    }
    return workCanvasRef.current.getContext("2d", { willReadFrequently: true })!;
  }, []);

  useEffect(() => {
    setModelStatus("loading");
    loadModel()
      .then(() => setModelStatus("ready"))
      .catch((err) => {
        console.error(err);
        setModelError(err?.message ?? "Failed to load model");
        setModelStatus("error");
      });
  }, []);

  useEffect(() => {
    if (!isLive) {
      runningRef.current = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      setDetections([]); setFps(0); setLatency(0); setElapsed(0);
      return;
    }
    runningRef.current = true;
    const start = Date.now();
    let frames = 0;
    let fpsTimer = Date.now();

    const loop = async () => {
      if (!runningRef.current) return;
      const video = webcamRef.current?.video as HTMLVideoElement | undefined;
      if (video && video.readyState === 4 && modelStatus === "ready") {
        try {
          const t0 = performance.now();
          const dets = await detect(video, getWorkCtx(), 0.4);
          const t1 = performance.now();
          setLatency(Math.round(t1 - t0));
          setDetections(dets);
          drawOverlay(dets, video.videoWidth, video.videoHeight);
        } catch (err) {
          console.error("inference error", err);
        }
      }
      frames++;
      const now = Date.now();
      if (now - fpsTimer >= 1000) {
        setFps(frames); frames = 0; fpsTimer = now;
        setElapsed(Math.floor((now - start) / 1000));
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => {
      runningRef.current = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [isLive, modelStatus, getWorkCtx]);

  const drawOverlay = (dets: Detection[], srcW: number, srcH: number) => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    if (canvas.width !== srcW || canvas.height !== srcH) {
      canvas.width = srcW; canvas.height = srcH;
    }
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, srcW, srcH);
    ctx.save();
    ctx.translate(srcW, 0);
    ctx.scale(-1, 1);
    ctx.lineWidth = Math.max(2, srcW / 320);
    ctx.font = `${Math.max(14, srcW / 60)}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textBaseline = "top";

    for (const d of dets) {
      const color = BOX_COLORS[d.classId % BOX_COLORS.length];
      const w = d.x2 - d.x1;
      const h = d.y2 - d.y1;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.strokeRect(d.x1, d.y1, w, h);
      const label = `${d.label} ${(d.score * 100).toFixed(0)}%`;
      const padding = 4;
      const metrics = ctx.measureText(label);
      const tagH = parseInt(ctx.font, 10) + padding * 2;
      ctx.fillRect(d.x1, Math.max(0, d.y1 - tagH), metrics.width + padding * 2, tagH);
      ctx.fillStyle = "#0a0a0a";
      ctx.save();
      ctx.translate(d.x1 + metrics.width + padding, Math.max(0, d.y1 - tagH) + padding);
      ctx.scale(-1, 1);
      ctx.fillText(label, 0, 0);
      ctx.restore();
    }
    ctx.restore();
  };

  const mm = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const ss = String(elapsed % 60).padStart(2, "0");

  const topDetections = (() => {
    const byLabel = new Map<string, { conf: number; classId: number; count: number }>();
    for (const d of detections) {
      const cur = byLabel.get(d.label);
      if (!cur || d.score > cur.conf) {
        byLabel.set(d.label, { conf: d.score, classId: d.classId, count: (cur?.count ?? 0) + 1 });
      } else {
        cur.count += 1;
      }
    }
    return Array.from(byLabel.entries()).sort((a, b) => b[1].conf - a[1].conf).slice(0, 6);
  })();

  return (
    <div className="min-h-screen bg-slate-950 text-white px-4 sm:px-8 lg:px-12 py-8 font-sans">
      <header className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 pb-6 mb-8 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="w-11 h-11 rounded-2xl bg-gradient-to-br from-cyan-400 to-violet-500 flex items-center justify-center shadow-lg shadow-cyan-500/30">
            <FiEye className="text-2xl text-slate-950" />
          </div>
          <div>
            <h1 className="text-2xl sm:text-3xl font-black tracking-tight">
              <span className="bg-gradient-to-r from-cyan-300 to-violet-400 bg-clip-text text-transparent">VISION</span>{" "}
              <span className="text-white">PRO AI</span>
            </h1>
            <p className="text-xs text-slate-400 tracking-wider uppercase">Realtime YOLOv8-n neural perception</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-2 rounded-xl border border-slate-800 bg-slate-900/60 backdrop-blur">
            <FiWifi className="text-cyan-400" />
            <span className="text-xs font-medium text-slate-300">
              {modelStatus === "ready" && "Model ready · WASM"}
              {modelStatus === "loading" && "Loading YOLOv8-n…"}
              {modelStatus === "error" && "Model error"}
              {modelStatus === "idle" && "Edge node · online"}
            </span>
          </div>
          <button
            onClick={() => setIsLive(!isLive)}
            disabled={modelStatus !== "ready"}
            className={`px-6 py-3 rounded-xl font-bold text-sm tracking-wide transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed ${
              isLive
                ? "bg-rose-600 text-white hover:opacity-90"
                : "bg-gradient-to-r from-cyan-400 to-violet-500 text-slate-950 hover:-translate-y-0.5 hover:shadow-lg hover:shadow-cyan-500/40"
            }`}
          >
            <span className="flex items-center gap-2">
              {isLive ? <FiVideoOff /> : <FiVideo />}
              {isLive ? "STOP FEED" : "START LIVE AI"}
            </span>
          </button>
        </div>
      </header>

      {modelStatus === "error" && (
        <div className="mb-6 p-4 rounded-xl border border-rose-500/40 bg-rose-500/10 text-sm text-rose-200">
          Failed to load YOLOv8-n model: {modelError}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-6">
        <div className="relative aspect-video rounded-3xl border border-slate-800 bg-black overflow-hidden">
          {["top-4 left-4 border-l-2 border-t-2","top-4 right-4 border-r-2 border-t-2","bottom-4 left-4 border-l-2 border-b-2","bottom-4 right-4 border-r-2 border-b-2"].map((c) => (
            <div key={c} className={`absolute w-8 h-8 border-cyan-400/70 z-20 pointer-events-none ${c}`} />
          ))}

          <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-2 px-4 py-1.5 rounded-full bg-black/60 backdrop-blur border border-slate-800">
            <span className={`w-2 h-2 rounded-full ${isLive ? "bg-rose-500 animate-pulse" : "bg-slate-500"}`} />
            <span className="text-[11px] font-semibold tracking-widest uppercase text-white">
              {isLive ? `REC · ${mm}:${ss}` : modelStatus === "loading" ? "Loading model" : "Standby"}
            </span>
          </div>

          {isLive ? (
            <>
              <Webcam
                ref={webcamRef}
                className="w-full h-full object-cover"
                screenshotFormat="image/jpeg"
                audio={false}
                mirrored
                videoConstraints={{ width: 1280, height: 720, facingMode: "user" }}
              />
              <canvas ref={overlayRef} className="absolute inset-0 w-full h-full pointer-events-none" />
            </>
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center text-slate-400 bg-gradient-to-br from-slate-900 to-slate-950">
              <div className="w-20 h-20 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center mb-5">
                <FiCpu className="text-4xl text-cyan-400 animate-pulse" />
              </div>
              <p className="text-sm tracking-[0.3em] uppercase">
                {modelStatus === "loading" ? "Loading YOLOv8-n…" : "System Standby"}
              </p>
              <p className="text-xs mt-2 text-slate-500">
                {modelStatus === "ready" ? <>Click <span className="text-cyan-400 font-semibold">START LIVE AI</span> to engage</>
                  : modelStatus === "error" ? <>Model unavailable — check console</>
                  : <>Fetching ~12 MB ONNX weights…</>}
              </p>
            </div>
          )}
        </div>

        <aside className="flex flex-col gap-4">
          <div className="grid grid-cols-2 gap-3">
            <Stat label="FPS" value={isLive ? `${fps}` : "—"} icon={FiActivity} accent />
            <Stat label="Latency" value={isLive ? `${latency} ms` : "—"} icon={FiZap} />
            <Stat label="Model" value="YOLOv8-n" icon={FiCpu} />
            <Stat label="Input" value={`${INPUT_SIZE}×${INPUT_SIZE}`} icon={FiCamera} />
          </div>

          <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs font-bold tracking-widest uppercase text-slate-400">Live Detections</h3>
              <span className="text-[10px] text-cyan-400 font-semibold">
                {isLive ? `${detections.length} OBJECTS` : "IDLE"}
              </span>
            </div>
            {topDetections.length === 0 ? (
              <p className="text-xs text-slate-500">
                {isLive ? "Scanning frame…" : "Start the feed to see real-time COCO classes."}
              </p>
            ) : (
              <ul className="space-y-3">
                {topDetections.map(([label, info]) => {
                  const color = BOX_COLORS[info.classId % BOX_COLORS.length];
                  return (
                    <li key={label}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="font-medium text-white capitalize">
                          {label}
                          {info.count > 1 && <span className="ml-1 text-slate-400">×{info.count}</span>}
                        </span>
                        <span className="text-slate-400">{(info.conf * 100).toFixed(0)}%</span>
                      </div>
                      <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
                        <div className="h-full transition-all duration-300"
                          style={{ width: `${info.conf * 100}%`, background: color }} />
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>

          <div className="bg-gradient-to-br from-cyan-400 to-violet-500 rounded-2xl p-5 text-slate-950">
            <p className="text-[10px] tracking-widest uppercase opacity-80">Neural Core</p>
            <p className="text-2xl font-black mt-1">YOLOv8-n · COCO</p>
            <p className="text-xs opacity-90 mt-2">
              ONNX Runtime Web · WASM backend · 80 classes · in-browser inference.
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}
