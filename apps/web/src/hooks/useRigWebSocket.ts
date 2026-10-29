"use client";

import { useEffect, useRef } from "react";
import {
    RigJobResponseSchema,
    WsTelemetryEventSchema,
} from "@chronorig/contracts";
import { useRigStore } from "../store/useRigStore";

interface UseRigWebSocketOptions {
    jobId: string;
    apiBaseUrl: string;
    wsBaseUrl: string;
}

export function useRigWebSocket({
    jobId,
    apiBaseUrl,
    wsBaseUrl,
}: UseRigWebSocketOptions): void {
    const applyTelemetryEvent = useRigStore((s) => s.applyTelemetryEvent);
    const setWsConnected = useRigStore((s) => s.setWsConnected);
    const setJob = useRigStore((s) => s.setJob);

    const retryCountRef = useRef(0);
    const wsRef = useRef<WebSocket | null>(null);

    useEffect(() => {
        let isUnmounted = false;
        let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

        const fetchLatestJobSnapshot = async () => {
            try {
                const res = await fetch(`${apiBaseUrl}/api/v1/jobs/${jobId}`);
                if (!res.ok) return;
                const raw: unknown = await res.json();
                const parsed = RigJobResponseSchema.safeParse(raw);
                if (parsed.success && !isUnmounted) {
                    setJob(parsed.data);
                }
            } catch (err) {
                console.warn("Failed to fetch initial RigJob snapshot:", err);
            }
        };

        const connectWebSocket = () => {
            if (isUnmounted) return;
            const socketUrl = `${wsBaseUrl}/api/v1/jobs/${jobId}/stream`;
            const ws = new WebSocket(socketUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                if (isUnmounted) return;
                retryCountRef.current = 0;
                setWsConnected(true);
            };

            ws.onmessage = (event: MessageEvent<string>) => {
                if (isUnmounted) return;
                try {
                    const rawJson: unknown = JSON.parse(event.data);
                    const validated = WsTelemetryEventSchema.safeParse(rawJson);
                    if (!validated.success) {
                        console.error(
                            "Dropped malformed WS telemetry frame:",
                            validated.error.flatten()
                        );
                        return;
                    }

                    applyTelemetryEvent(validated.data);

                    if (
                        validated.data.event === "STATUS_CHANGED" &&
                        validated.data.status === "COMPLETED"
                    ) {
                        void fetchLatestJobSnapshot();
                    }
                } catch (err) {
                    console.error("Failed to parse WebSocket JSON payload:", err);
                }
            };

            ws.onclose = () => {
                if (isUnmounted) return;
                setWsConnected(false);
                const attempt = Math.min(retryCountRef.current + 1, 6);
                retryCountRef.current = attempt;
                const backoffMs = Math.min(1000 * 2 ** attempt + Math.random() * 400, 15000);
                reconnectTimer = setTimeout(connectWebSocket, backoffMs);
            };

            ws.onerror = () => {
                ws.close();
            };
        };

        void fetchLatestJobSnapshot();
        connectWebSocket();

        return () => {
            isUnmounted = true;
            if (reconnectTimer) clearTimeout(reconnectTimer);
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, [jobId, apiBaseUrl, wsBaseUrl, applyTelemetryEvent, setWsConnected, setJob]);
}