BEGIN;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TYPE job_status_enum AS ENUM (
    'QUEUED',
    'SEGMENTING_SAM',
    'SOLVING_KINEMATICS',
    'TEXTURING_MUSE_IMAGE',
    'SYNTHESIZING_PHYSICS',
    'VERIFYING_IOU',
    'COMPLETED',
    'FAILED'
);

CREATE TYPE joint_type_enum AS ENUM (
    'fixed',
    'revolute',
    'prismatic',
    'spherical'
);

CREATE TABLE rig_job (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    owner_id VARCHAR(128) NOT NULL,
    status job_status_enum NOT NULL DEFAULT 'QUEUED',
    video_s3_key VARCHAR(512) NOT NULL,
    video_fps DOUBLE PRECISION NOT NULL DEFAULT 30.0,
    video_frame_count INTEGER NOT NULL DEFAULT 0,
    part_prompts JSONB NOT NULL DEFAULT '[]'::jsonb,
    kinematic_summary JSONB,
    generated_r3f_code TEXT,
    generated_bpy_s3_key VARCHAR(512),
    final_iou_score DOUBLE PRECISION,
    error_code VARCHAR(64),
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX ix_rig_job_owner_status_created ON rig_job (owner_id, status, created_at DESC);
CREATE INDEX ix_rig_job_kinematic_gin ON rig_job USING GIN (kinematic_summary);

CREATE TABLE rigid_body_part (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES rig_job(id) ON DELETE CASCADE,
    sam_object_id INTEGER NOT NULL,
    label VARCHAR(128) NOT NULL,
    is_static_root BOOLEAN NOT NULL DEFAULT FALSE,
    estimated_mass_kg DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    bounding_box_extents JSONB NOT NULL,
    trajectory_frames JSONB NOT NULL,
    texture_s3_key VARCHAR(512),
    UNIQUE (job_id, sam_object_id)
);

CREATE INDEX ix_rigid_body_part_job_id ON rigid_body_part (job_id);

CREATE TABLE kinematic_joint (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES rig_job(id) ON DELETE CASCADE,
    parent_part_id UUID NOT NULL REFERENCES rigid_body_part(id) ON DELETE CASCADE,
    child_part_id UUID NOT NULL REFERENCES rigid_body_part(id) ON DELETE CASCADE,
    joint_type joint_type_enum NOT NULL,
    anchor_world_xyz JSONB NOT NULL,
    axis_xyz JSONB NOT NULL,
    limits_rad_or_m JSONB NOT NULL,
    CONSTRAINT chk_distinct_joint_parts CHECK (parent_part_id <> child_part_id)
);

CREATE INDEX ix_kinematic_joint_job_id ON kinematic_joint (job_id);

CREATE TABLE iou_verification_iteration (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES rig_job(id) ON DELETE CASCADE,
    iteration_number INTEGER NOT NULL,
    mean_iou DOUBLE PRECISION NOT NULL CHECK (mean_iou >= 0.0 AND mean_iou <= 1.0),
    frame_iou_breakdown JSONB NOT NULL,
    agent_diff_patch TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (job_id, iteration_number)
);

CREATE INDEX ix_iou_verification_job_id ON iou_verification_iteration (job_id, iteration_number);

COMMIT;