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

    // Sync video time -> frame index
    useEffect(() => {
        const video = videoRef.current;
        if (!video || !job) return;

        let animFrameId: number;
        const updateFrameLoop = () => {
            if (!video.paused && !video.ended) {
                const frameIdx = Math.min(
                    Math.floor(video.currentTime * job.videoFps),
                    Math.max(0, job.videoFrameCount - 1)
                );
                setCurrentFrameIndex(frameIdx);
            }
            animFrameId = requestAnimationFrame(updateFrameLoop);
        };

        animFrameId = requestAnimationFrame(updateFrameLoop);
        return () => cancelAnimationFrame(animFrameId);
    }, [job, setCurrentFrameIndex]);

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

            // Translucent SAM 3.1 mask bounding region
            ctx.save();
            ctx.strokeStyle = color;
            ctx.fillStyle = `${color}22`;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.roundRect(bx, by, bw, bh, 4);
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

        // 2. Render Kinematic Joint Anchors (Projected from World XYZ to 2D Canvas Space)
        if (showJointAnchors) {
            job.joints.forEach((joint) => {
                // Map normalized world [-2..2] coordinates to 2D overlay center
                const px = width * 0.5 + joint.anchorWorldXyz[0] * (width * 0.2);
                const py = height * 0.5 - joint.anchorWorldXyz[1] * (height * 0.2);

                ctx.save();
                ctx.strokeStyle = "#FEE440";
                ctx.fillStyle = "#111827";
                ctx.lineWidth = 2;

                ctx.beginPath();
                ctx.arc(px, py, 7, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();

                ctx.fillStyle = "#FEE440";
                ctx.font = "bold 10px ui-monospace, SFMono-Regular, monospace";
                ctx.fillText(joint.jointType.toUpperCase(), px + 10, py + 3);
                ctx.restore();
            });
        }
    }, [job, currentFrameIndex, showPcaVectors, showJointAnchors]);

    const handlePlayPause = () => {
        const video = videoRef.current;
        if (!video) return;
        if (video.paused) {
            void video.play();
            setIsPlaying(true);
        } else {
            video.pause();
            setIsPlaying(false);
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
                        className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-mono transition ${showPcaVectors
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
                        className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-mono transition ${showJointAnchors
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
                    className="p-1.5 rounded bg-teal-500 hover:bg-teal-400 text-slate-950 transition"
                    aria-label={isPlaying ? "Pause video" : "Play video"}
                >
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </button>

                <input
                    type="range"
                    min={0}
                    max={Math.max(1, (job?.videoFrameCount ?? 60) - 1)}
                    value={currentFrameIndex}
                    onChange={(e) => {
                        const frame = Number(e.target.value);
                        setCurrentFrameIndex(frame);
                        if (videoRef.current && job) {
                            videoRef.current.currentTime = frame / job.videoFps;
                        }
                    }}
                    className="flex-1 accent-teal-400 cursor-pointer"
                />

                <span className="text-xs font-mono text-slate-400 min-w-[92px] text-right">
                    Frame {currentFrameIndex} / {job?.videoFrameCount ?? 0}
                </span>
            </div>
        </div>
    );
};