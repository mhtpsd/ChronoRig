import { z } from "zod";

const EnvSchema = z.object({
    NODE_ENV: z.enum(["development", "test", "production"]),
    NEXT_PUBLIC_API_BASE_URL: z.string().url(),
    NEXT_PUBLIC_WS_BASE_URL: z
        .string()
        .regex(/^wss?:\/\/.+/, "Must be a valid ws:// or wss:// URL"),
    MUSE_CODE_SANDBOX_TIMEOUT_MS: z.coerce.number().int().min(5000).max(120000).default(30000),
});

export const validateEnv = () => {
    const parsed = EnvSchema.safeParse(process.env);
    if (!parsed.success) {
        console.error(
            "FATAL: Invalid environment variables detected at startup:",
            JSON.stringify(parsed.error.flatten().fieldErrors, null, 2)
        );
        process.exit(1);
    }
    return parsed.data;
};