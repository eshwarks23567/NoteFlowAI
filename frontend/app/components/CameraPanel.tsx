"use client";
import { useRef, useState, useEffect, useCallback } from "react";

interface CameraPanelProps {
    isActive: boolean;
    onFrameCapture?: (frameData: string) => void;
}

export default function CameraPanel({ isActive, onFrameCapture }: CameraPanelProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);

    const [cameraActive, setCameraActive] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [facingMode, setFacingMode] = useState<"user" | "environment">("environment");
    const [audioLevel, setAudioLevel] = useState(0);
    const [frameCount, setFrameCount] = useState(0);
    const [isManuallyStopped, setIsManuallyStopped] = useState(false);

    const startCamera = useCallback(async () => {
        try {
            setError(null);
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: facingMode,
                    width: { ideal: 1920 },
                    height: { ideal: 1080 },
                    frameRate: { ideal: 30 },
                },
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                },
            });

            streamRef.current = stream;
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                try {
                    await videoRef.current.play();
                } catch (playError: any) {
                    if (playError.name !== "AbortError") {
                        console.error("Video play error:", playError);
                    }
                }
            }
            setCameraActive(true);

            // Audio level monitoring
            const audioCtx = new AudioContext();
            const analyser = audioCtx.createAnalyser();
            const source = audioCtx.createMediaStreamSource(stream);
            source.connect(analyser);
            analyser.fftSize = 256;
            const dataArray = new Uint8Array(analyser.frequencyBinCount);

            const updateAudio = () => {
                if (!streamRef.current) return;
                analyser.getByteFrequencyData(dataArray);
                const avg = dataArray.reduce((a, b) => a + b) / dataArray.length;
                setAudioLevel(Math.min(100, (avg / 128) * 100));
                requestAnimationFrame(updateAudio);
            };
            updateAudio();

            // Frame capture every 3rd second (for processing)
            frameIntervalRef.current = setInterval(() => {
                captureFrame();
            }, 3000);

        } catch (err: any) {
            console.error("Camera error:", err);
            if (err.name === "NotAllowedError") {
                setError("Camera access denied. Please allow camera permissions.");
            } else if (err.name === "NotFoundError") {
                setError("No camera found. Please connect a webcam.");
            } else {
                setError(`Camera error: ${err.message}`);
            }
        }
    }, [facingMode]);

    const stopCamera = useCallback(() => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
        if (frameIntervalRef.current) {
            clearInterval(frameIntervalRef.current);
        }
        setCameraActive(false);
        setAudioLevel(0);
        setFrameCount(0);
    }, []);

    const captureFrame = useCallback(() => {
        if (!videoRef.current || !canvasRef.current) return;
        const video = videoRef.current;
        const canvas = canvasRef.current;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(video, 0, 0);
        const frameData = canvas.toDataURL("image/jpeg", 0.7);
        setFrameCount(prev => prev + 1);
        onFrameCapture?.(frameData);
    }, [onFrameCapture]);

    const toggleFacing = () => {
        const newFacing = facingMode === "user" ? "environment" : "user";
        setFacingMode(newFacing);
        if (cameraActive) {
            stopCamera();
            setTimeout(() => startCamera(), 300);
        }
    };

    // Auto-start camera when session becomes active
    useEffect(() => {
        if (isActive && !cameraActive && !error && !isManuallyStopped) {
            startCamera();
        } else if (!isActive && cameraActive) {
            stopCamera();
        }
    }, [isActive, cameraActive, startCamera, stopCamera, error, isManuallyStopped]);

    // Reset manually stopped state when a new session starts
    useEffect(() => {
        if (isActive) {
            setIsManuallyStopped(false);
        }
    }, [isActive]);

    // Stop camera on unmount
    useEffect(() => {
        return () => stopCamera();
    }, [stopCamera]);

    return (
        <div className="glass-card camera-panel">
            <div className="panel-header">
                <span className="panel-title">
                    <span className="panel-title-icon">CAM</span>
                    Live Camera Feed
                </span>
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                    {cameraActive && (
                        <span className="status-badge live" style={{ padding: "3px 10px", fontSize: 10 }}>
                            <span className="pulse-dot" style={{ width: 6, height: 6 }} />
                            REC
                        </span>
                    )}
                    {cameraActive && (
                        <span style={{
                            fontFamily: "'JetBrains Mono', monospace",
                            fontSize: 10,
                            color: "var(--text-muted)",
                        }}>
                            {frameCount} frames
                        </span>
                    )}
                </div>
            </div>

            <div className="camera-body">
                {/* Video + Canvas */}
                <div className="camera-viewport">
                    <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className={`camera-video ${cameraActive ? "active" : ""}`}
                    />
                    <canvas ref={canvasRef} style={{ display: "none" }} />

                    {/* Overlay when not active */}
                    {!cameraActive && !error && (
                        <div className="camera-placeholder">
                            <div className="camera-placeholder-icon">CAM</div>
                            <div className="camera-placeholder-text">
                                Point your camera at the projection screen & professor
                            </div>
                            <div className="camera-placeholder-sub">
                                Dual-stream: slide content + professor behavior
                            </div>
                        </div>
                    )}

                    {/* Error state */}
                    {error && (
                        <div className="camera-error">
                            <span style={{ fontSize: 28 }}>!</span>
                            <span>{error}</span>
                        </div>
                    )}

                    {/* Audio level bar overlay */}
                    {cameraActive && (
                        <div className="camera-audio-overlay">
                            <div className="audio-label">MIC</div>
                            <div className="audio-bar-track">
                                <div
                                    className="audio-bar-fill"
                                    style={{
                                        width: `${audioLevel}%`,
                                        background: audioLevel > 70
                                            ? "var(--color-critical)"
                                            : audioLevel > 30
                                                ? "var(--color-important)"
                                                : "var(--color-supporting)",
                                    }}
                                />
                            </div>
                            <div className="audio-level-text">{Math.round(audioLevel)}%</div>
                        </div>
                    )}

                    {/* Processing indicator */}
                    {cameraActive && isActive && (
                        <div className="camera-processing-badge">
                            <span className="processing-dot" /> Processing every 3s
                        </div>
                    )}
                </div>

                {/* Controls */}
                <div className="camera-controls">
                    {!cameraActive ? (
                        <button className="btn btn-primary camera-btn" onClick={() => { setIsManuallyStopped(false); startCamera(); }}>
                            Start Camera
                        </button>
                    ) : (
                        <button className="btn btn-danger camera-btn" onClick={() => { setIsManuallyStopped(true); stopCamera(); }}>
                            Stop Camera
                        </button>
                    )}
                    <button
                        className="btn btn-ghost camera-btn-sm"
                        onClick={toggleFacing}
                        title="Switch between front and rear camera"
                    >
                        {facingMode === "user" ? "Front" : "Rear"}
                    </button>
                    {cameraActive && (
                        <button
                            className="btn btn-ghost camera-btn-sm"
                            onClick={captureFrame}
                            title="Capture current frame for OCR"
                        >
                            Snap
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
}
