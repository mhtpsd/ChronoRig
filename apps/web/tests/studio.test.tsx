import { describe, expect, it, beforeEach } from "vitest";
import {
    CreateRigJobRequestSchema,
    RigJobResponseSchema,
    WsTelemetryEventSchema,
} from "@chronorig/contracts";
import { useRigStore } from "../src/store/useRigStore";
import { computeLocalAnchorOffset } from "../src/components/canvas/PhysicsRigViewport";

const MOCK_JOB = {
    id: "11111111-2222-4333-8444-555555555555",
    status: "QUEUED" as const,
    videoUrl: "http://localhost:8000/api/v1/jobs/11111111-2222-4333-8444-555555555555/video.mp4",
    videoFps: 30,
    videoFrameCount: 36,
    partPrompts: ["base", "arm"],
    parts: [],
    joints: [],
    iouIterations: [],
    finalIouScore: null,
    generatedR3fCode: null,
    blenderScriptDownloadUrl: null,
    errorCode: null,
    errorMessage: null,
    createdAt: "2026-11-02T10:00:00.000Z",
    updatedAt: "2026-11-02T10:00:00.000Z",
};

describe("ChronoRig Web Studio Frontend Guardrails", () => {
    beforeEach(() => {
        useRigStore.setState({
            job: null,
            currentFrameIndex: 0,
            isPlaying: false,
            isSilhouetteMode: false,
            showPcaVectors: true,
            showJointAnchors: true,
            wsConnected: false,
            progressPct: 0,
            statusMessage: "Waiting",
            telemetryLogs: [],
        });
    });

    it("computes accurate local-space Rapier joint anchor offsets", () => {
        expect(computeLocalAnchorOffset([0.2, 1.5, -0.25], [0.1, 1.2, -0.25])).toEqual([
            0.1, 0.3, 0,
        ]);
    });

    it("handles full Zustand store lifecycle and all 4 WebSocket telemetry event variants", () => {
        const parsedJob = RigJobResponseSchema.parse(MOCK_JOB);
        useRigStore.getState().setJob(parsedJob);
        expect(useRigStore.getState().progressPct).toBe(15);

        useRigStore.getState().setCurrentFrameIndex(12);
        useRigStore.getState().setIsPlaying(true);
        useRigStore.getState().setSilhouetteMode(true);
        useRigStore.getState().togglePcaVectors();
        useRigStore.getState().toggleJointAnchors();
        useRigStore.getState().setWsConnected(true);

        expect(useRigStore.getState().currentFrameIndex).toBe(12);
        expect(useRigStore.getState().isPlaying).toBe(true);
        expect(useRigStore.getState().isSilhouetteMode).toBe(true);
        expect(useRigStore.getState().showPcaVectors).toBe(false);
        expect(useRigStore.getState().showJointAnchors).toBe(false);
        expect(useRigStore.getState().wsConnected).toBe(true);

        // 1. STATUS_CHANGED
        useRigStore.getState().applyTelemetryEvent(
            WsTelemetryEventSchema.parse({
                event: "STATUS_CHANGED",
                jobId: MOCK_JOB.id,
                status: "SOLVING_KINEMATICS",
                progressPct: 50,
                message: "Solving PCA...",
            })
        );
        expect(useRigStore.getState().job?.status).toBe("SOLVING_KINEMATICS");

        // 2. SAM_FRAME_MASK_READY
        useRigStore.getState().applyTelemetryEvent(
            WsTelemetryEventSchema.parse({
                event: "SAM_FRAME_MASK_READY",
                jobId: MOCK_JOB.id,
                frameIndex: 5,
                objectCount: 2,
            })
        );
        expect(useRigStore.getState().statusMessage).toContain("frame #5");

        // 3. IOU_ITERATION_COMPLETED + appendIouIteration update
        const iter1 = {
            iterationNumber: 1,
            meanIou: 0.71,
            frameIouBreakdown: { "0": 0.71 },
            agentDiffPatch: "patch-1",
            createdAt: "2026-11-02T10:00:05.000Z",
        };
        useRigStore.getState().applyTelemetryEvent(
            WsTelemetryEventSchema.parse({
                event: "IOU_ITERATION_COMPLETED",
                jobId: MOCK_JOB.id,
                iteration: iter1,
            })
        );
        expect(useRigStore.getState().job?.finalIouScore).toBe(0.71);

        useRigStore.getState().appendIouIteration({ ...iter1, meanIou: 0.92 });
        expect(useRigStore.getState().job?.finalIouScore).toBe(0.92);

        // 4. JOB_FAILED
        useRigStore.getState().applyTelemetryEvent(
            WsTelemetryEventSchema.parse({
                event: "JOB_FAILED",
                jobId: MOCK_JOB.id,
                errorCode: "SAM_TIMEOUT",
                errorMessage: "Timed out",
            })
        );
        expect(useRigStore.getState().job?.status).toBe("FAILED");
        expect(useRigStore.getState().job?.errorCode).toBe("SAM_TIMEOUT");

        // 5. Validate CreateRigJobRequestSchema defaults
        const req = CreateRigJobRequestSchema.parse({
            videoS3Key: "uploads/test.mp4",
            partPrompts: ["base", "arm"],
        });
        expect(req.targetIouThreshold).toBe(0.85);
        expect(req.maxSelfDebugIterations).toBe(3);
    });
});