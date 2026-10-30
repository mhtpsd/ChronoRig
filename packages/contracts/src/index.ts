import { z } from "zod";

export const JobStatusSchema = z.enum([
    "QUEUED",
    "SEGMENTING_SAM",
    "SOLVING_KINEMATICS",
    "TEXTURING_MUSE_IMAGE",
    "SYNTHESIZING_PHYSICS",
    "VERIFYING_IOU",
    "COMPLETED",
    "FAILED",
]);
export type JobStatus = z.infer<typeof JobStatusSchema>;

export const JointTypeSchema = z.enum([
    "fixed",
    "revolute",
    "prismatic",
    "spherical",
]);
export type JointType = z.infer<typeof JointTypeSchema>;

export const Vector3Schema = z.tuple([z.number(), z.number(), z.number()]);

export const FramePoseSchema = z.object({
    frameIndex: z.number().int().nonnegative(),
    timestampSec: z.number().nonnegative(),
    centroid2dPx: z.tuple([z.number(), z.number()]),
    angleRad: z.number(),
    angleConfidence: z.enum(["high", "low_eccentricity"]).optional(),
    bboxPx: z.tuple([z.number(), z.number(), z.number(), z.number()]),
    contourPolygonPx: z.array(z.tuple([z.number(), z.number()])).optional(),
});

export const RigidBodyPartSchema = z.object({
    id: z.string().uuid(),
    samObjectId: z.number().int(),
    label: z.string().min(1).max(128),
    isStaticRoot: z.boolean(),
    estimatedMassKg: z.number().positive(),
    boundingBoxExtents: Vector3Schema,
    trajectoryFrames: z.array(FramePoseSchema),
    textureUrl: z.string().url().nullable(),
});

export const KinematicJointSchema = z.object({
    id: z.string().uuid(),
    parentPartId: z.string().uuid(),
    childPartId: z.string().uuid(),
    jointType: JointTypeSchema,
    anchorWorldXyz: Vector3Schema,
    axisXyz: Vector3Schema,
    limits: z.object({
        min: z.number(),
        max: z.number(),
        stiffness: z.number().nonnegative().default(0),
        damping: z.number().nonnegative().default(0.5),
    }),
});

export const IouIterationSchema = z.object({
    iterationNumber: z.number().int().positive(),
    meanIou: z.number().min(0).max(1),
    frameIouBreakdown: z.record(z.string(), z.number().min(0).max(1)),
    agentDiffPatch: z.string().nullable(),
    createdAt: z.string().datetime(),
});

export const CreateRigJobRequestSchema = z.object({
    videoS3Key: z.string().min(5).max(512),
    partPrompts: z.array(z.string().min(1).max(64)).min(2).max(12),
    targetIouThreshold: z.number().min(0.5).max(0.98).default(0.85),
    maxSelfDebugIterations: z.number().int().min(1).max(5).default(3),
});
export type CreateRigJobRequest = z.infer<typeof CreateRigJobRequestSchema>;

export const RigJobResponseSchema = z.object({
    id: z.string().uuid(),
    status: JobStatusSchema,
    videoUrl: z.string().url(),
    videoFps: z.number().positive(),
    videoFrameCount: z.number().int().nonnegative(),
    partPrompts: z.array(z.string()),
    parts: z.array(RigidBodyPartSchema),
    joints: z.array(KinematicJointSchema),
    iouIterations: z.array(IouIterationSchema),
    finalIouScore: z.number().min(0).max(1).nullable(),
    generatedR3fCode: z.string().nullable(),
    blenderScriptDownloadUrl: z.string().url().nullable(),
    errorCode: z.string().nullable(),
    errorMessage: z.string().nullable(),
    createdAt: z.string().datetime(),
    updatedAt: z.string().datetime(),
});
export type RigJobResponse = z.infer<typeof RigJobResponseSchema>;

export const WsTelemetryEventSchema = z.discriminatedUnion("event", [
    z.object({
        event: z.literal("STATUS_CHANGED"),
        jobId: z.string().uuid(),
        status: JobStatusSchema,
        progressPct: z.number().min(0).max(100),
        message: z.string(),
    }),
    z.object({
        event: z.literal("SAM_FRAME_MASK_READY"),
        jobId: z.string().uuid(),
        frameIndex: z.number().int().nonnegative(),
        objectCount: z.number().int().nonnegative(),
    }),
    z.object({
        event: z.literal("IOU_ITERATION_COMPLETED"),
        jobId: z.string().uuid(),
        iteration: IouIterationSchema,
    }),
    z.object({
        event: z.literal("JOB_FAILED"),
        jobId: z.string().uuid(),
        errorCode: z.string(),
        errorMessage: z.string(),
    }),
]);
export type WsTelemetryEvent = z.infer<typeof WsTelemetryEventSchema>;