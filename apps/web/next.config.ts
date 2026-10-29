import type { NextConfig } from "next";

const nextConfig: NextConfig = {
    reactStrictMode: true,
    transpilePackages: ["three", "@chronorig/contracts"],
    webpack: (config: { experiments: any; }) => {
        config.experiments = {
            ...config.experiments,
            asyncWebAssembly: true,
            layers: true,
        };
        return config;
    },
};

export default nextConfig;