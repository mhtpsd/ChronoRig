"use client";

import React, { createRef, useMemo, useRef } from "react";
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
import type { KinematicJointSchema, RigidBodyPartSchema } from "@chronorig/contracts";
import type { z } from "zod";
import { Box, ScanEye, Sparkles } from "lucide-react";
import { useRigStore } from "../../store/useRigStore";

type RigidBodyPart = z.infer<typeof RigidBodyPartSchema>;
type KinematicJoint = z.infer<typeof KinematicJointSchema>;

/**
 * Computes the local anchor offset of a world-space joint pivot relative to a part's initial world position.
 */
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
    parentRef: React.RefObject<RapierRigidBody>;
    childRef: React.RefObject<RapierRigidBody>;
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
    useRevoluteJoint(parentRef, childRef, [
        localAnchorA,
        localAnchorB,
        joint.axisXyz,
    ]);
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
    usePrismaticJoint(parentRef, childRef, [
        localAnchorA,
        localAnchorB,
        joint.axisXyz,
        [joint.limits.min, joint.limits.max],
    ]);
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
    useFixedJoint(parentRef, childRef, [
        localAnchorA,
        [0, 0, 0, 1],
        localAnchorB,
        [0, 0, 0, 1],
    ]);
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
    useSphericalJoint(parentRef, childRef, [localAnchorA, localAnchorB]);
    return null;
};

interface DynamicSceneProps {
    parts: RigidBodyPart[];
    joints: KinematicJoint[];
    isSilhouetteMode: boolean;
}

const DynamicPhysicsScene: React.FC<DynamicSceneProps> = ({
    parts,
    joints,
    isSilhouetteMode,
}) => {
    const bodyRefs = useMemo(() => {
        const map = new Map<string, React.RefObject<RapierRigidBody>>();
        parts.forEach((p) => {
            map.set(p.id, createRef<RapierRigidBody>());
        });
        return map;
    }, [parts]);

    // Derive deterministic initial 3D coordinates from frame 0 centroid & boundingBoxExtents
    const partOrigins = useMemo(() => {
        const map = new Map<string, [number, number, number]>();
        parts.forEach((p, idx) => {
            const frame0 = p.trajectoryFrames[0];
            const normX = frame0 ? (frame0.centroid2dPx[0] - 320) / 140 : idx * 0.6;
            const normY = frame0 ? (240 - frame0.centroid2dPx[1]) / 140 + 1.2 : 1.5;
            map.set(p.id, [Number(normX.toFixed(3)), Number(normY.toFixed(3)), 0]);
        });
        return map;
    }, [parts]);

    return (
        <Physics gravity={[0, -9.81, 0]}>
            {parts.map((part, idx) => {
                const ref = bodyRefs.get(part.id);
                const origin = partOrigins.get(part.id) ?? [0, 1.5, 0];
                const [wx, wy, wz] = part.boundingBoxExtents;

                // Mitigation #3: Place all articulated rig parts into group 1 with filter 2
                // so directly jointed parent/child rigid bodies never self-collide on Frame 0
                const collisionGroupMask = interactionGroups(1, [2]);

                return (
                    <RigidBody
                        key={part.id}
                        ref={ref}
                        type={part.isStaticRoot ? "fixed" : "dynamic"}
                        position={origin}
                        mass={part.estimatedMassKg}
                        collisionGroups={collisionGroupMask}
                        linearDamping={0.4}
                        angularDamping={0.4}
                        onClick={(e) => {
                            e.stopPropagation();
                            if (!part.isStaticRoot && ref?.current) {
                                // Interactive test impulse when user clicks a part in the studio
                                ref.current.applyImpulse({ x: 0.4, y: 1.2, z: 0 }, true);
                            }
                        }}
                    >
                        <CuboidCollider args={[wx / 2, wy / 2, wz / 2]} />
                        <mesh castShadow={!isSilhouetteMode} receiveShadow={!isSilhouetteMode}>
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

            {/* Ground Plane (Group 2 so it collides with Group 1 parts) */}
            <RigidBody
                type="fixed"
                position={[0, -0.25, 0]}
                collisionGroups={interactionGroups(2, [1])}
            >
                <CuboidCollider args={[6, 0.25, 6]} />
            </RigidBody>

            {/* Bind all KinematicJoint constraints */}
            {joints.map((joint) => {
                const parentRef = bodyRefs.get(joint.parentPartId);
                const childRef = bodyRefs.get(joint.childPartId);
                const parentOrigin = partOrigins.get(joint.parentPartId) ?? [0, 0, 0];
                const childOrigin = partOrigins.get(joint.childPartId) ?? [0, 0, 0];

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
    const isSilhouetteMode = useRigStore((s) => s.isSilhouetteMode);
    const setSilhouetteMode = useRigStore((s) => s.setSilhouetteMode);
    const canvasWrapperRef = useRef<HTMLDivElement | null>(null);

    return (
        <div
            ref={canvasWrapperRef}
            className="flex flex-col h-full bg-slate-950 border-r border-slate-800 relative"
        >
            {/* Top Toolbar */}
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-slate-800 bg-slate-900/60 z-10">
                <div className="flex items-center gap-2">
                    <Box className="w-4 h-4 text-rose-400" />
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                        React Three Fiber + Rapier 3D Rig
                    </span>
                </div>

                <div className="flex items-center gap-2">
                    <button
                        type="button"
                        onClick={() => setSilhouetteMode(!isSilhouetteMode)}
                        className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono transition ${isSilhouetteMode
                            ? "bg-white text-slate-950 font-bold"
                            : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                            }`}
                    >
                        <ScanEye className="w-3.5 h-3.5" />
                        {isSilhouetteMode ? "Binary Silhouette (IoU Mode)" : "PBR Lit Mode"}
                    </button>
                </div>
            </div>

            {/* Interactive 3D WebGL2 Canvas */}
            <div className="flex-1 relative">
                <Canvas
                    shadows
                    camera={{ position: [0, 1.8, 4.5], fov: 45 }}
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
                            isSilhouetteMode={isSilhouetteMode}
                        />
                    )}

                    <OrbitControls makeDefault />
                </Canvas>

                {/* Interactive Hint Overlay */}
                <div className="absolute bottom-3 left-3 pointer-events-none flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-900/80 border border-slate-800 text-[11px] font-mono text-slate-400">
                    <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                    Click any dynamic rigid body to apply a physics test impulse
                </div>
            </div>
        </div>
    );
};