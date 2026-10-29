import { create } from "zustand";
import type {
    IouIterationSchema,
    JobStatus,
    RigJobResponse,
    WsTelemetryEvent,
} from "@chronorig/contracts";
import type { z } from "zod";

type IouIteration = z.infer<typeof IouIterationSchema>;

export interface TelemetryLogEntry {
    id: string;
    timestamp: string;
    event: WsTelemetryEvent["event"];
    summary: string;
}

interface RigStudioState {
    job: RigJobResponse | null;
    currentFrameIndex: number;
    isPlaying: boolean;
    isSilhouetteMode: boolean;
    showPcaVectors: boolean;
    showJointAnchors: boolean;
    wsConnected: boolean;
    progressPct: number;
    statusMessage: string;
    telemetryLogs: TelemetryLogEntry[];

    // Actions
    setJob: (job: RigJobResponse) => void;
    setCurrentFrameIndex: (frameIndex: number) => void;
    setIsPlaying: (playing: boolean) => void;
    setSilhouetteMode: (enabled: boolean) => void;
    togglePcaVectors: () => void;
    toggleJointAnchors: () => void;
    setWsConnected: (connected: boolean) => void;
    applyTelemetryEvent: (payload: WsTelemetryEvent) => void;
    appendIouIteration: (iteration: IouIteration) => void;
}

export const useRigStore = create<RigStudioState>((set) => ({
    job: null,
    currentFrameIndex: 0,
    isPlaying: false,
    isSilhouetteMode: false,
    showPcaVectors: true,
    showJointAnchors: true,
    wsConnected: false,
    progressPct: 0,
    statusMessage: "Waiting for pipeline telemetry...",
    telemetryLogs: [],

    setJob: (job) =>
        set({
            job,
            progressPct: job.status === "COMPLETED" ? 100 : 15,
            statusMessage: `Loaded job in state ${job.status}`,
        }),

    setCurrentFrameIndex: (frameIndex) => set({ currentFrameIndex: frameIndex }),
    setIsPlaying: (playing) => set({ isPlaying: playing }),
    setSilhouetteMode: (enabled) => set({ isSilhouetteMode: enabled }),
    togglePcaVectors: () => set((s) => ({ showPcaVectors: !s.showPcaVectors })),
    toggleJointAnchors: () => set((s) => ({ showJointAnchors: !s.showJointAnchors })),
    setWsConnected: (connected) => set({ wsConnected: connected }),

    appendIouIteration: (iteration) =>
        set((state) => {
            if (!state.job) return state;
            const existing = state.job.iouIterations.some(
                (it) => it.iterationNumber === iteration.iterationNumber
            );
            const updatedIterations = existing
                ? state.job.iouIterations.map((it) =>
                    it.iterationNumber === iteration.iterationNumber ? iteration : it
                )
                : [...state.job.iouIterations, iteration];

            return {
                job: {
                    ...state.job,
                    iouIterations: updatedIterations,
                    finalIouScore: iteration.meanIou,
                },
            };
        }),

    applyTelemetryEvent: (payload) =>
        set((state) => {
            const nowIso = new Date().toISOString();
            const makeLog = (summary: string): TelemetryLogEntry => ({
                id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
                timestamp: nowIso,
                event: payload.event,
                summary,
            });

            switch (payload.event) {
                case "STATUS_CHANGED": {
                    const nextJob = state.job
                        ? { ...state.job, status: payload.status as JobStatus }
                        : null;
                    return {
                        job: nextJob,
                        progressPct: payload.progressPct,
                        statusMessage: payload.message,
                        telemetryLogs: [
                            makeLog(`[${payload.status}] ${payload.message} (${payload.progressPct}%)`),
                            ...state.telemetryLogs.slice(0, 99),
                        ],
                    };
                }

                case "SAM_FRAME_MASK_READY": {
                    return {
                        statusMessage: `SAM 3.1 segmented frame #${payload.frameIndex} (${payload.objectCount} rigid parts tracked)`,
                        telemetryLogs: [
                            makeLog(
                                `SAM 3.1 decoded frame ${payload.frameIndex}: ${payload.objectCount} masks`
                            ),
                            ...state.telemetryLogs.slice(0, 99),
                        ],
                    };
                }

                case "IOU_ITERATION_COMPLETED": {
                    const iter = payload.iteration;
                    const updatedJob = state.job
                        ? {
                            ...state.job,
                            finalIouScore: iter.meanIou,
                            iouIterations: [
                                ...state.job.iouIterations.filter(
                                    (i) => i.iterationNumber !== iter.iterationNumber
                                ),
                                iter,
                            ],
                        }
                        : null;

                    return {
                        job: updatedJob,
                        statusMessage: `Muse Code Iteration #${iter.iterationNumber} verified: Mean IoU ${(
                            iter.meanIou * 100
                        ).toFixed(1)}%`,
                        telemetryLogs: [
                            makeLog(
                                `IoU Iteration #${iter.iterationNumber} -> ${(iter.meanIou * 100).toFixed(1)}%`
                            ),
                            ...state.telemetryLogs.slice(0, 99),
                        ],
                    };
                }

                case "JOB_FAILED": {
                    const failedJob = state.job
                        ? {
                            ...state.job,
                            status: "FAILED" as JobStatus,
                            errorCode: payload.errorCode,
                            errorMessage: payload.errorMessage,
                        }
                        : null;
                    return {
                        job: failedJob,
                        statusMessage: `Error [${payload.errorCode}]: ${payload.errorMessage}`,
                        telemetryLogs: [
                            makeLog(`FATAL [${payload.errorCode}]: ${payload.errorMessage}`),
                            ...state.telemetryLogs.slice(0, 99),
                        ],
                    };
                }
            }
        }),
}));