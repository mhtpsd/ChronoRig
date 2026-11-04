"use client";

import React, { useEffect, useRef } from "react";
import { Eye, Compass, Crosshair, Play, Pause } from "lucide-react";
import { useRigStore } from "../../store/useRigStore";

const PART_NEON_PALETTE = [
    "#00F5D4", // Cyan Neon
    "#FF2A85", // Magenta Neon
    "#FEE440", // Cyber Yellow
    "#9B5DE5", // Electric Purple
    "#00BBF9", // Sky Neon
    "#F15BB5", // Hot Pink
];

export const SamMaskOverlayCanvas: React.FC = () => {
    const job = useRigStore((s) => s.job);
    const currentFrameIndex = useRigStore((s) => s.currentFrameIndex);
    const isPlaying = useRigStore((s) => s.isPlaying);
    const showPcaVectors = useRigStore((s) => s.showPcaVectors);
    const showJointAnchors = useRigStore((s) => s.showJointAnchors);

    const setCurrentFrameIndex = useRigStore((s) => s.setCurrentFrameIndex);
    const setIsPlaying = useRigStore((s) => s.setIsPlaying);
    const togglePcaVectors = useRigStore((s) => s.togglePcaVectors);
    const toggleJointAnchors = useRigStore((s) => s.toggleJointAnchors);

    const videoRef = useRef<HTMLVideoElement | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    const totalFrames = Math.max(1, job?.videoFrameCount ?? 90);
    const fps = (job?.videoFps && job.videoFps > 0) ? job.videoFps : 30;

    // Synchronize HTML5 video playback -> frame index
    useEffect(() => {
        const video = videoRef.current;
        if (!video || !job) return;

        let animFrameId: number;
        const updateFrameLoop = () => {
            if (!video.paused && !video.ended) {
                const calculatedFrame = Math.min(
                    Math.floor(video.currentTime * fps),
                    totalFrames - 1
                );
                setCurrentFrameIndex(calculatedFrame);
            }
            animFrameId = requestAnimationFrame(updateFrameLoop);
        };

        animFrameId = requestAnimationFrame(updateFrameLoop);
        return () => cancelAnimationFrame(animFrameId);
    }, [job, fps, totalFrames, setCurrentFrameIndex]);

    // Fallback animation ticker when video element is not available or blocked
    useEffect(() => {
        if (!isPlaying) return;
        const video = videoRef.current;
        if (video && !video.paused && !video.ended) return;

        const intervalMs = Math.round(1000 / fps);
        const timer = setInterval(() => {
            setCurrentFrameIndex((useRigStore.getState().currentFrameIndex + 1) % totalFrames);
        }, intervalMs);

        return () => clearInterval(timer);
    }, [isPlaying, fps, totalFrames, setCurrentFrameIndex]);

    // Draw SAM 3.1 bounding boxes, centroid trails, PCA orientation vectors & joint pivots
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !job) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const width = canvas.width;
        const height = canvas.height;
        ctx.clearRect(0, 0, width, height);

        // 1. Render each tracked RigidBodyPart at currentFrameIndex
        job.parts.forEach((part, idx) => {
            const color = PART_NEON_PALETTE[idx % PART_NEON_PALETTE.length] ?? "#00F5D4";
            const pose =
                part.trajectoryFrames.find((f) => f.frameIndex === currentFrameIndex) ??
                part.trajectoryFrames[0];

            if (!pose) return;

            const [bx, by, bw, bh] = pose.bboxPx;
            const [cx, cy] = pose.centroid2dPx;
            const polygon = pose.contourPolygonPx;

            // Draw actual SAM 3.1 / OpenCV polygon mask contour if present, fallback to bbox
            ctx.save();
            ctx.strokeStyle = color;
            ctx.fillStyle = `${color}33`;
            ctx.lineWidth = 2.2;
            ctx.beginPath();
            if (polygon && polygon.length >= 3) {
                polygon.forEach(([px, py], pIdx) => {
                    if (pIdx === 0) ctx.moveTo(px, py);
                    else ctx.lineTo(px, py);
                });
                ctx.closePath();
            } else {
                ctx.roundRect(bx, by, bw, bh, 4);
            }
            ctx.fill();
            ctx.stroke();

            // Label badge
            ctx.fillStyle = color;
            ctx.font = "bold 11px ui-monospace, SFMono-Regular, Menlo, monospace";
            ctx.fillText(
                `#${part.samObjectId} ${part.label} (${part.isStaticRoot ? "ROOT" : `${part.estimatedMassKg}kg`})`,
                bx + 4,
                Math.max(14, by - 6)
            );

            // Centroid history trajectory trail
            ctx.beginPath();
            ctx.strokeStyle = `${color}88`;
            ctx.lineWidth = 1.5;
            part.trajectoryFrames
                .filter((f) => f.frameIndex <= currentFrameIndex)
                .slice(-25)
                .forEach((histPose, hIdx) => {
                    const [hx, hy] = histPose.centroid2dPx;
                    if (hIdx === 0) ctx.moveTo(hx, hy);
                    else ctx.lineTo(hx, hy);
                });
            ctx.stroke();

            // PCA Principal Orientation Vector (theta_t)
            if (showPcaVectors) {
                const vecLen = 42;
                const ex = cx + Math.cos(pose.angleRad) * vecLen;
                const ey = cy + Math.sin(pose.angleRad) * vecLen;

                ctx.beginPath();
                ctx.strokeStyle = "#FFFFFF";
                ctx.lineWidth = 2.5;
                ctx.moveTo(cx, cy);
                ctx.lineTo(ex, ey);
                ctx.stroke();

                ctx.beginPath();
                ctx.fillStyle = color;
                ctx.arc(cx, cy, 4.5, 0, Math.PI * 2);
                ctx.fill();
            }

            ctx.restore();
        });

        // 2. Render Kinematic Joint Anchors (Correct Projection from World XYZ to 2D Canvas Space)
        if (showJointAnchors && job.joints) {
            job.joints.forEach((joint) => {
                // Exact inverse projection of world coordinates to 2D pixel coordinates:
                // world_x = (px - (width / 2.0)) / 140.0 => px = world_x * 140.0 + (width / 2.0)
                // world_y = ((height / 2.0) - py) / 140.0 + 1.2 => py = (height / 2.0) - (world_y - 1.2) * 140.0
                const px = joint.anchorWorldXyz[0] * 140.0 + width * 0.5;
                const py = height * 0.5 - (joint.anchorWorldXyz[1] - 1.2) * 140.0;

                ctx.save();
                ctx.strokeStyle = "#FEE440";
                ctx.fillStyle = "#111827";
                ctx.lineWidth = 2.5;

                // Outer joint anchor ring
                ctx.beginPath();
                ctx.arc(px, py, 7.5, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();

                // Inner crosshair marking the exact pivot center
                ctx.beginPath();
                ctx.strokeStyle = "#FEE440";
                ctx.lineWidth = 1.5;
                ctx.moveTo(px - 4, py);
                ctx.lineTo(px + 4, py);
                ctx.moveTo(px, py - 4);
                ctx.lineTo(px, py + 4);
                ctx.stroke();

                // Joint type badge
                ctx.fillStyle = "#FEE440";
                ctx.font = "bold 10px ui-monospace, SFMono-Regular, monospace";
                ctx.fillText(joint.jointType.toUpperCase(), px + 11, py + 3.5);
                ctx.restore();
            });
        }
    }, [job, currentFrameIndex, showPcaVectors, showJointAnchors]);

    const handlePlayPause = () => {
        const video = videoRef.current;
        if (!video) {
            setIsPlaying(!isPlaying);
            return;
        }

        if (isPlaying) {
            video.pause();
            setIsPlaying(false);
        } else {
            if (video.ended || (video.duration && video.currentTime >= video.duration - 0.1)) {
                video.currentTime = 0;
            }
            video.play()
                .then(() => setIsPlaying(true))
                .catch((err) => {
                    console.warn("Video play interrupted/blocked, running timer playback loop:", err);
                    setIsPlaying(true);
                });
        }
    };

    return (
        <div className="flex flex-col h-full bg-slate-950 border-r border-slate-800 select-none">
            {/* Header Bar */}
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-slate-800 bg-slate-900/60">
                <div className="flex items-center gap-2">
                    <Eye className="w-4 h-4 text-teal-400" />
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                        SAM 3.1 Spatiotemporal Perception
                    </span>
                </div>
                <div className="flex items-center gap-1.5">
                    <button
                        type="button"
                        onClick={togglePcaVectors}
                        className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-mono transition ${
                            showPcaVectors
                                ? "bg-teal-500/20 text-teal-300 border border-teal-500/40"
                                : "bg-slate-800 text-slate-400"
                        }`}
                    >
                        <Compass className="w-3.5 h-3.5" />
                        PCA θ
                    </button>
                    <button
                        type="button"
                        onClick={toggleJointAnchors}
                        className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-mono transition ${
                            showJointAnchors
                                ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                                : "bg-slate-800 text-slate-400"
                        }`}
                    >
                        <Crosshair className="w-3.5 h-3.5" />
                        Pivots
                    </button>
                </div>
            </div>

            {/* Video + Synchronized 2D Mask Canvas */}
            <div className="relative flex-1 flex items-center justify-center bg-black overflow-hidden">
                {job?.videoUrl ? (
                    <video
                        ref={videoRef}
                        src={job.videoUrl}
                        className="w-full h-full object-contain"
                        loop
                        muted
                        playsInline
                        onPlay={() => setIsPlaying(true)}
                        onPause={() => setIsPlaying(false)}
                        onEnded={() => setIsPlaying(false)}
                    />
                ) : (
                    <div className="text-xs font-mono text-slate-500">
                        Awaiting input_video stream...
                    </div>
                )}
                <canvas
                    ref={canvasRef}
                    width={640}
                    height={480}
                    className="absolute inset-0 w-full h-full pointer-events-none object-contain"
                />
            </div>

            {/* Timeline Scrubber Footer */}
            <div className="px-4 py-3 border-t border-slate-800 bg-slate-900/80 flex items-center gap-3">
                <button
                    type="button"
                    onClick={handlePlayPause}
                    className="p-1.5 rounded bg-teal-500 hover:bg-teal-400 text-slate-950 transition cursor-pointer"
                    aria-label={isPlaying ? "Pause video" : "Play video"}
                >
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </button>

                <input
                    type="range"
                    min={0}
                    max={totalFrames - 1}
                    value={currentFrameIndex}
                    onChange={(e) => {
                        const frame = Number(e.target.value);
                        setCurrentFrameIndex(frame);
                        if (videoRef.current && Number.isFinite(videoRef.current.duration)) {
                            videoRef.current.currentTime = frame / fps;
                        }
                    }}
                    className="flex-1 accent-teal-400 cursor-pointer"
                />

                <span className="text-xs font-mono text-slate-400 min-w-[92px] text-right">
                    Frame {currentFrameIndex + 1} / {totalFrames}
                </span>
            </div>
        </div>
    );
};
