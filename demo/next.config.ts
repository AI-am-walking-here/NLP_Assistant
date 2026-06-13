import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // /health and /api/systems use App Router proxies (graceful 503 while API boots).
};

export default nextConfig;
