import { describe, expect, it, beforeEach } from "vitest";
import { WsTelemetryEventSchema } from "@chronorig/contracts";
import { useRigStore } from "../src/store/useRigStore";
import { computeLocalAnchorOffset } from "../src/components/canvas/PhysicsRigViewport";

describe("ChronoRig Web Studio Frontend Guardrails", () => {
    beforeEach(() => {
        useRigStore.setState({
            job: null,
            currentFrameIndex: 0,
            isPlaying: false,
            isSilhouetteMode: false,
            wsConnected: false,
            progressPct: 0,
            telemetryLogs: [],
        });
    });

    it("computes accurate local-space Rapier joint anchor offsets", () => {
        const worldAnchor: [number, number, number] = [0.20, 1.50, 0.0];
        const bodyOrigin: [number, number, number] = [0.10, 1.20, 0.0];

        const localOffset = computeLocalAnchorOffset(worldAnchor, bodyOrigin);
        expect(localOffset).toEqual([0.1, 0.3, 0]);
    });

    it("validates and applies STATUS_CHANGED telemetry events in store", () => {
        const rawEvent = {
            event: "STATUS_CHANGED",
            jobId: "11111111-2222-4333-8444-555555555555",
            status: "SOLVING_KINEMATICS",
            progressPct: 50,
            message: "Solving PCA orientations and hinge pivots...",
        };

        const parsed = WsTelemetryEventSchema.safeParse(rawEvent);
        expect(parsed.success).toBe(true);

        if (parsed.success) {
            useRigStore.getState().applyTelemetryEvent(parsed.data);
        }

        const state = useRigStore.getState();
        expect(state.progressPct).toBe(50);
        expect(state.statusMessage).toContain("Solving PCA");
        expect(state.telemetryLogs.length).toBe(1);
    });
});