"use client";

import React from "react";
import {
    Activity,
    CheckCircle2,
    Code2,
    Download,
    Terminal,
    Wifi,
    WifiOff,
} from "lucide-react";
import { useRigStore } from "../../store/useRigStore";

const PIPELINE_STEPS = [
    { key: "SEGMENTING_SAM", label: "SAM 3.1 Spatiotemporal Masks" },
    { key: "SOLVING_KINEMATICS", label: "OpenCV PCA & Hinge Solver" },
    { key: "TEXTURING_MUSE_IMAGE", label: "Muse Image Texture Inpainting" },
    { key: "SYNTHESIZING_PHYSICS", label: "Muse Spark 1.3 + Muse Code Rig" },
    { key: "VERIFYING_IOU", label: "Closed-Loop Silhouette IoU Check" },
    { key: "COMPLETED", label: "Verified Rig Ready" },
] as const;

export const TelemetryPanel: React.FC = () => {
    const job = useRigStore((s) => s.job);
    const wsConnected = useRigStore((s) => s.wsConnected);
    const progressPct = useRigStore((s) => s.progressPct);
    const statusMessage = useRigStore((s) => s.statusMessage);
    const telemetryLogs = useRigStore((s) => s.telemetryLogs);

    const iterations = job?.iouIterations ?? [];
    const latestDiff =
        iterations.length > 0
            ? iterations[iterations.length - 1]?.agentDiffPatch
            : null;

    return (
        <div className="flex flex-col h-full bg-slate-950 text-slate-200 overflow-y-auto">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-slate-800 bg-slate-900/60">
                <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-purple-400" />
                    <span className="text-xs font-semibold uppercase tracking-wider">
                        Muse Code Self-Debugging Telemetry
                    </span>
                </div>

                <span
                    className={`flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono ${wsConnected
                            ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                            : "bg-rose-500/15 text-rose-400 border border-rose-500/30"
                        }`}
                >
                    {wsConnected ? (
                        <>
                            <Wifi className="w-3 h-3" /> LIVE WS
                        </>
                    ) : (
                        <>
                            <WifiOff className="w-3 h-3" /> RECONNECTING
                        </>
                    )}
                </span>
            </div>

            {/* Pipeline Progress Bar & Stepper */}
            <div className="p-4 border-b border-slate-800 space-y-3">
                <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-teal-300 truncate">{statusMessage}</span>
                    <span className="text-slate-400">{progressPct}%</span>
                </div>
                <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                        className="h-full bg-gradient-to-r from-teal-400 via-purple-400 to-rose-400 transition-all duration-300"
                        style={{ width: `${progressPct}%` }}
                    />
                </div>

                <div className="grid grid-cols-2 gap-1.5 pt-1">
                    {PIPELINE_STEPS.map((step) => {
                        const isCurrent = job?.status === step.key;
                        const isDone = job?.status === "COMPLETED";
                        return (
                            <div
                                key={step.key}
                                className={`flex items-center gap-1.5 px-2 py-1.5 rounded text-[11px] font-mono border ${isCurrent
                                        ? "bg-purple-500/20 border-purple-500/50 text-purple-200"
                                        : isDone
                                            ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
                                            : "bg-slate-900/60 border-slate-800 text-slate-400"
                                    }`}
                            >
                                <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                                <span className="truncate">{step.label}</span>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Silhouette IoU Convergence Curve */}
            <div className="p-4 border-b border-slate-800 space-y-2">
                <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                        Silhouette IoU Accuracy vs. SAM 3.1
                    </span>
                    <span className="text-sm font-mono font-bold text-teal-400">
                        {job?.finalIouScore
                            ? `${(job.finalIouScore * 100).toFixed(1)}%`
                            : "Pending"}
                    </span>
                </div>

                <div className="h-28 w-full bg-slate-900/90 rounded border border-slate-800 p-3 flex items-end gap-4 relative">
                    {/* 85% Target IoU Threshold Line */}
                    <div
                        className="absolute left-0 right-0 border-t border-dashed border-amber-400/60 pointer-events-none"
                        style={{ bottom: "85%" }}
                    >
                        <span className="absolute right-2 -top-4 text-[10px] font-mono text-amber-300">
                            Target 85% IoU
                        </span>
                    </div>

                    {iterations.length === 0 ? (
                        <div className="text-xs font-mono text-slate-500 m-auto">
                            Waiting for Iteration #1 silhouette render...
                        </div>
                    ) : (
                        iterations.map((iter) => {
                            const heightPct = Math.round(iter.meanIou * 100);
                            const passed = iter.meanIou >= 0.85;
                            return (
                                <div
                                    key={iter.iterationNumber}
                                    className="flex-1 flex flex-col items-center gap-1 h-full justify-end"
                                >
                                    <span className="text-[11px] font-mono font-bold text-slate-200">
                                        {heightPct}%
                                    </span>
                                    <div
                                        className={`w-full max-w-[44px] rounded-t transition-all ${passed ? "bg-emerald-400" : "bg-purple-500"
                                            }`}
                                        style={{ height: `${heightPct}%` }}
                                    />
                                    <span className="text-[10px] font-mono text-slate-400">
                                        Iter #{iter.iterationNumber}
                                    </span>
                                </div>
                            );
                        })
                    )}
                </div>
            </div>

            {/* Latest @muse-code/sdk Self-Debug Patch */}
            <div className="p-4 border-b border-slate-800 space-y-2">
                <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400">
                    <Code2 className="w-3.5 h-3.5 text-teal-400" />
                    <span>Latest @muse-code/sdk Joint Patch</span>
                </div>
                <pre className="p-2.5 rounded bg-slate-900 border border-slate-800 text-[11px] font-mono text-teal-300 overflow-x-auto max-h-36">
                    {latestDiff ??
                        "// No joint offset patch required yet — awaiting verification loop."}
                </pre>
            </div>

            {/* Live WebSocket Event Stream */}
            <div className="p-4 flex-1 flex flex-col min-h-[160px] space-y-2">
                <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400">
                    <Terminal className="w-3.5 h-3.5 text-purple-400" />
                    <span>Live Pipeline Event Stream</span>
                </div>
                <div className="flex-1 rounded bg-black/80 border border-slate-800 p-2.5 font-mono text-[11px] space-y-1.5 overflow-y-auto max-h-48">
                    {telemetryLogs.map((log) => (
                        <div key={log.id} className="text-slate-300 leading-relaxed">
                            <span className="text-slate-500">
                                [{log.timestamp.slice(11, 19)}]
                            </span>{" "}
                            <span className="text-teal-400">{log.event}:</span> {log.summary}
                        </div>
                    ))}
                </div>
            </div>

            {/* Export Blender bpy Footer */}
            <div className="p-4 border-t border-slate-800 bg-slate-900/60">
                <a
                    href={job?.blenderScriptDownloadUrl ?? "#"}
                    download={`chronorig-${job?.id ?? "scene"}.py`}
                    className="w-full flex items-center justify-center gap-2 py-2 px-4 rounded bg-gradient-to-r from-teal-500 to-emerald-500 hover:from-teal-400 hover:to-emerald-400 text-slate-950 font-semibold text-xs transition"
                >
                    <Download className="w-4 h-4" />
                    Export Articulated Blender (bpy) Script
                </a>
            </div>
        </div>
    );
};