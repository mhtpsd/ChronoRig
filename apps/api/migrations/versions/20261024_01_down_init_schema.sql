BEGIN;

DROP TABLE IF EXISTS iou_verification_iteration;
DROP TABLE IF EXISTS kinematic_joint;
DROP TABLE IF EXISTS rigid_body_part;
DROP TABLE IF EXISTS rig_job;

DROP TYPE IF EXISTS joint_type_enum;
DROP TYPE IF EXISTS job_status_enum;

COMMIT;