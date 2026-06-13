import type { NextConfig } from "next";

/**
 * FPM consent-plane frontend config.
 *
 * Rewrites: the FPM FastAPI service (auth + dashboard JSON API) is proxied
 * same-origin so the httpOnly session cookie "just works" with no CORS. The
 * dev server runs on 3002; the backend on FPM_API_BASE (default :8090).
 *
 * For real Google sign-in, register the redirect URI as
 * http://localhost:3002/auth/callback and set FPM_OAUTH_REDIRECT_URI to match.
 */
const apiBase = process.env.FPM_API_BASE ?? "http://localhost:8090";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      { source: "/v1/:path*", destination: `${apiBase}/v1/:path*` },
      { source: "/auth/:path*", destination: `${apiBase}/auth/:path*` },
      { source: "/health", destination: `${apiBase}/health` },
    ];
  },
};

export default nextConfig;
