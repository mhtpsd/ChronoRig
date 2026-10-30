"use client";

import React, { createRef, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Grid } from "@react-three/drei";
import {
    Physics,
    RigidBody,
    CuboidCollider,
    interactionGroups,
    useRevoluteJoint,
    usePrismaticJoint,
    useFixedJoint,
    useSphericalJoint,
    type RapierRigidBody,
} from "@react-three/rapier";
import {
    RigJobResponseSchema,
    type KinematicJointSchema,
    type RigidBodyPartSchema,
} from "@chronorig/contracts";
import type { z } from "zod";
import { Box, PlayCircle, ScanEye, Sparkles, Cpu } from "lucide-react";
import { useRigStore } from "../../store/useRigStore";

type RigidBodyPart = z.infer<typeof RigidBodyPartSchema>;
type KinematicJoint = z.infer<typeof KinematicJointSchema>;

export function computeLocalAnchorOffset(
    worldAnchor: [number, number, number],
    bodyWorldOrigin: [number, number, number]
): [number, number, number] {
    return [
        Number((worldAnchor[0] - bodyWorldOrigin[0]).toFixed(4)),
        Number((worldAnchor[1] - bodyWorldOrigin[1]).toFixed(4)),
        Number((worldAnchor[2] - bodyWorldOrigin[2]).toFixed(4)),
    ];
}

interface JointBinderProps {
    joint: KinematicJoint;
    parentRef: React.RefObject<RapierRigidBody | null>;
    childRef: React.RefObject<RapierRigidBody | null>;
    parentOrigin: [number, number, number];
    childOrigin: [number, number, number];
}

const RevoluteJointBinder: React.FC<JointBinderProps> = ({
    joint,
    parentRef,
    childRef,
    parentOrigin,
    childOrigin,
}) => {
    const localAnchorA = computeLocalAnchorOffset(joint.anchorWorldXyz, parentOrigin);
    const localAnchorB = computeLocalAnchorOffset(joint.anchorWorldXyz, childOrigin);
    useRevoluteJoint(
        parentRef as React.RefObject<RapierRigidBody>,
        childRef as React.RefObject<RapierRigidBody>,
        [localAnchorA, localAnchorB, joint.axisXyz]
    );
    return null;
};

const PrismaticJointBinder: React.FC<JointBinderProps> = ({
    joint,
    parentRef,
    childRef,
    parentOrigin,
    childOrigin,
}) => {
    const localAnchorA = computeLocalAnchorOffset(joint.anchorWorldXyz, parentOrigin);
    const localAnchorB = computeLocalAnchorOffset(joint.anchorWorldXyz, childOrigin);
    usePrismaticJoint(
        parentRef as React.RefObject<RapierRigidBody>,
        childRef as React.RefObject<RapierRigidBody>,
        [localAnchorA, localAnchorB, joint.axisXyz, [joint.limits.min, joint.limits.max]]
    );
    return null;
};

const FixedJointBinder: React.FC<JointBinderProps> = ({
    joint,
    parentRef,
    childRef,
    parentOrigin,
    childOrigin,
}) => {
    const localAnchorA = computeLocalAnchorOffset(joint.anchorWorldXyz, parentOrigin);
    const localAnchorB = computeLocalAnchorOffset(joint.anchorWorldXyz, childOrigin);
    useFixedJoint(
        parentRef as React.RefObject<RapierRigidBody>,
        childRef as React.RefObject<RapierRigidBody>,
        [localAnchorA, [0, 0, 0, 1], localAnchorB, [0, 0, 0, 1]]
    );
    return null;
};

const SphericalJointBinder: React.FC<JointBinderProps> = ({
    joint,
    parentRef,
    childRef,
    parentOrigin,
    childOrigin,
}) => {
    const localAnchorA = computeLocalAnchorOffset(joint.anchorWorldXyz, parentOrigin);
    const localAnchorB = computeLocalAnchorOffset(joint.anchorWorldXyz, childOrigin);
    useSphericalJoint(
        parentRef as React.RefObject<RapierRigidBody>,
        childRef as React.RefObject<RapierRigidBody>,
        [localAnchorA, localAnchorB]
    );
    return null;
};

interface DynamicSceneProps {
    parts: RigidBodyPart[];
    joints: KinematicJoint[];
    currentFrameIndex: number;
    isPoseReplayMode: boolean;
    isSilhouetteMode: boolean;
}

const DynamicPhysicsScene: React.FC<DynamicSceneProps> = ({
    parts,
    joints,
    currentFrameIndex,
    isPoseReplayMode,
    isSilhouetteMode,
}) => {
    const bodyRefs = useMemo(() => {
        const map = new Map<string, React.RefObject<RapierRigidBody | null>>();
        parts.forEach((p) => {
            map.set(p.id, createRef<RapierRigidBody>());
        });
        return map;
    }, [parts]);

    const partTransforms = useMemo(() => {
        const map = new Map<
            string,
            { pos: [number, number, number]; rot: [number, number, number] }
        >();
        parts.forEach((p, idx) => {
            const targetFrame = isPoseReplayMode ? currentFrameIndex : 0;
            const pose =
                p.trajectoryFrames.find((f) => f.frameIndex === targetFrame) ??
                p.trajectoryFrames[0];
            const normX = pose ? (pose.centroid2dPx[0] - 320) / 140 : idx * 0.6;
            const normY = pose ? (240 - pose.centroid2dPx[1]) / 140 + 1.2 : 1.5;
            const rotZ = pose ? -pose.angleRad : 0;
            map.set(p.id, {
                pos: [Number(normX.toFixed(3)), Number(normY.toFixed(3)), 0],
                rot: [0, 0, Number(rotZ.toFixed(4))],
            });
        });
        return map;
    }, [parts, currentFrameIndex, isPoseReplayMode]);

    // Pose Replay Mode: deterministically position & orient meshes from observed video poses
    if (isPoseReplayMode) {
        return (
            <group>
                {parts.map((part, idx) => {
                    const tf = partTransforms.get(part.id) ?? {
                        pos: [0, 1.5, 0],
                        rot: [0, 0, 0],
                    };
                    const [wx, wy, wz] = part.boundingBoxExtents;
                    return (
                        <mesh
                            key={part.id}
                            position={tf.pos}
                            rotation={tf.rot}
                            castShadow={!isSilhouetteMode}
                            receiveShadow={!isSilhouetteMode}
                        >
                            <boxGeometry args={[wx, wy, wz]} />
                            {isSilhouetteMode ? (
                                <meshBasicMaterial color="#FFFFFF" toneMapped={false} />
                            ) : (
                                <meshStandardMaterial
                                    color={
                                        part.isStaticRoot
                                            ? "#475569"
                                            : idx % 2 === 0
                                                ? "#14B8A6"
                                                : "#F43F5E"
                                    }
                                    roughness={0.35}
                                    metalness={0.6}
                                />
                            )}
                        </mesh>
                    );
                })}
                {joints.map((joint) => (
                    <mesh key={joint.id} position={joint.anchorWorldXyz}>
                        <sphereGeometry args={[0.07, 16, 16]} />
                        <meshBasicMaterial color="#FEE440" />
                    </mesh>
                ))}
            </group>
        );
    }

    // Live Rapier Physics Dynamics Mode
    return (
        <Physics gravity={[0, -9.81, 0]}>
            {parts.map((part, idx) => {
                const ref = bodyRefs.get(part.id);
                const tf = partTransforms.get(part.id) ?? {
                    pos: [0, 1.5, 0],
                    rot: [0, 0, 0],
                };
                const [wx, wy, wz] = part.boundingBoxExtents;
                const collisionGroupMask = interactionGroups(1, [2]);

                return (
                    <RigidBody
                        key={part.id}
                        ref={ref}
                        type={part.isStaticRoot ? "fixed" : "dynamic"}
                        position={tf.pos}
                        rotation={tf.rot}
                        mass={part.estimatedMassKg}
                        collisionGroups={collisionGroupMask}
                        linearDamping={0.4}
                        angularDamping={0.4}
                    >
                        <CuboidCollider args={[wx / 2, wy / 2, wz / 2]} />
                        <mesh
                            castShadow={!isSilhouetteMode}
                            receiveShadow={!isSilhouetteMode}
                            onClick={(e) => {
                                e.stopPropagation();
                                if (!part.isStaticRoot && ref?.current) {
                                    ref.current.applyImpulse({ x: 0.4, y: 1.2, z: 0 }, true);
                                }
                            }}>
                            <boxGeometry args={[wx, wy, wz]} />
                            {isSilhouetteMode ? (
                                <meshBasicMaterial color="#FFFFFF" toneMapped={false} />
                            ) : (
                                <meshStandardMaterial
                                    color={
                                        part.isStaticRoot
                                            ? "#475569"
                                            : idx % 2 === 0
                                                ? "#14B8A6"
                                                : "#F43F5E"
                                    }
                                    roughness={0.35}
                                    metalness={0.6}
                                />
                            )}
                        </mesh>
                    </RigidBody>
                );
            })}

            <RigidBody
                type="fixed"
                position={[0, -0.25, 0]}
                collisionGroups={interactionGroups(2, [1])}
            >
                <CuboidCollider args={[6, 0.25, 6]} />
            </RigidBody>

            {joints.map((joint) => {
                const parentRef = bodyRefs.get(joint.parentPartId);
                const childRef = bodyRefs.get(joint.childPartId);
                const parentOrigin = partTransforms.get(joint.parentPartId)?.pos ?? [0, 0, 0];
                const childOrigin = partTransforms.get(joint.childPartId)?.pos ?? [0, 0, 0];

                if (!parentRef || !childRef) return null;

                const commonProps: JointBinderProps = {
                    joint,
                    parentRef,
                    childRef,
                    parentOrigin,
                    childOrigin,
                };

                switch (joint.jointType) {
                    case "revolute":
                        return <RevoluteJointBinder key={joint.id} {...commonProps} />;
                    case "prismatic":
                        return <PrismaticJointBinder key={joint.id} {...commonProps} />;
                    case "spherical":
                        return <SphericalJointBinder key={joint.id} {...commonProps} />;
                    case "fixed":
                    default:
                        return <FixedJointBinder key={joint.id} {...commonProps} />;
                }
            })}
        </Physics>
    );
};

export const PhysicsRigViewport: React.FC = () => {
    const job = useRigStore((s) => s.job);
    const currentFrameIndex = useRigStore((s) => s.currentFrameIndex);
    const isSilhouetteMode = useRigStore((s) => s.isSilhouetteMode);
    const setSilhouetteMode = useRigStore((s) => s.setSilhouetteMode);
    const setJob = useRigStore((s) => s.setJob);

    const [isPoseReplayMode, setIsPoseReplayMode] = useState<boolean>(true);
    const [isRunningBackend, setIsRunningBackend] = useState<boolean>(false);
    const canvasWrapperRef = useRef<HTMLDivElement | null>(null);

    const handleRunLiveBackendReconstruction = async () => {
        setIsRunningBackend(true);
        try {
            const apiBase =
                process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
            const res = await fetch(`${apiBase}/api/v1/jobs/demo-hinge`, {
                method: "POST",
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const raw: unknown = await res.json();
            const parsed = RigJobResponseSchema.safeParse(raw);
            if (parsed.success) {
                setJob(parsed.data);
            }
        } catch (err) {
            console.error("Live backend reconstruction failed:", err);
        } finally {
            setIsRunningBackend(false);
        }
    };

    return (
        <div
            ref={canvasWrapperRef}
            className="flex flex-col h-full bg-slate-950 border-r border-slate-800 relative"
        >
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-slate-800 bg-slate-900/60 z-10">
                <div className="flex items-center gap-2">
                    <Box className="w-4 h-4 text-rose-400" />
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                        3D Articulated Rig Studio
                    </span>
                </div>

                <div className="flex items-center gap-1.5">
                    <button
                        type="button"
                        onClick={() => void handleRunLiveBackendReconstruction()}
                        disabled={isRunningBackend}
                        className="flex items-center gap-1 px-2.5 py-1 rounded text-xs font-mono bg-teal-500 hover:bg-teal-400 text-slate-950 font-semibold transition disabled:opacity-50"
                    >
                        <Cpu className="w-3.5 h-3.5" />
                        {isRunningBackend ? "Solving..." : "Run Live Pipeline"}
                    </button>

                    <button
                        type="button"
                        onClick={() => setIsPoseReplayMode(!isPoseReplayMode)}
                        className={`flex items-center gap-1 px-2.5 py-1 rounded text-xs font-mono transition ${isPoseReplayMode
                                ? "bg-purple-500/20 text-purple-300 border border-purple-500/40"
                                : "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                            }`}
                    >
                        <PlayCircle className="w-3.5 h-3.5" />
                        {isPoseReplayMode ? "Pose Replay Mode" : "Live Rapier Physics"}
                    </button>

                    <button
                        type="button"
                        onClick={() => setSilhouetteMode(!isSilhouetteMode)}
                        className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-mono transition ${isSilhouetteMode
                                ? "bg-white text-slate-950 font-bold"
                                : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                            }`}
                    >
                        <ScanEye className="w-3.5 h-3.5" />
                        {isSilhouetteMode ? "Silhouette" : "Lit"}
                    </button>
                </div>
            </div>

            <div className="flex-1 relative">
                <Canvas
                    shadows
                    camera={{ position: [0, 1.5, 4.2], fov: 45 }}
                    gl={{ preserveDrawingBuffer: true, antialias: true }}
                    style={{ background: isSilhouetteMode ? "#000000" : "#090D16" }}
                >
                    {!isSilhouetteMode && (
                        <>
                            <ambientLight intensity={0.6} />
                            <directionalLight position={[5, 8, 5]} intensity={1.4} castShadow />
                            <Grid
                                infiniteGrid
                                fadeDistance={20}
                                sectionColor="#1e293b"
                                cellColor="#0f172a"
                            />
                        </>
                    )}

                    {job && job.parts.length > 0 && (
                        <DynamicPhysicsScene
                            parts={job.parts}
                            joints={job.joints}
                            currentFrameIndex={currentFrameIndex}
                            isPoseReplayMode={isPoseReplayMode}
                            isSilhouetteMode={isSilhouetteMode}
                        />
                    )}

                    <OrbitControls makeDefault />
                </Canvas>

                <div className="absolute bottom-3 left-3 pointer-events-none flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-900/80 border border-slate-800 text-[11px] font-mono text-slate-400">
                    <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                    {isPoseReplayMode
                        ? "Pose Replay: Scrub video timeline to inspect reconstructed 3D motion & yellow hinge anchors"
                        : "Live Physics: Click any dynamic body to apply a physical impulse under gravity"}
                </div>
            </div>
        </div>
    );
};