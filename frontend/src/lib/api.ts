/**
 * Client for the FPM consent-plane API. All paths are same-origin (proxied to
 * the FastAPI service by next.config rewrites), so the httpOnly session cookie
 * rides along automatically.
 */

export type UsageEvent = {
  event: string;
  consumer: string;
  purpose: string;
  ts: string;
};

export type Voiceprint = {
  workspace_id: string;
  voiceprint_id: string;
  name: string | null;
  owner_email: string;
  enroll_allowed: boolean;
  identify_allowed: boolean;
  enroll_count: number;
  quality_score: number;
  created_at: string;
  last_seen_at: string;
  usage: UsageEvent[];
};

export type Me = {
  email: string | null;
  signed_in: boolean;
  google_enabled: boolean;
  dev_login: boolean;
};

export type Proposal = {
  proposal_id: string;
  workspace_id: string;
  voiceprint_id: string;
  proposed_by: string;
  proposed_name: string;
  created_at: string;
};

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(path, { credentials: "same-origin" });
  if (!res.ok) throw new ApiError(res.status, `${res.status}`);
  return (await res.json()) as T;
}

async function postJSON<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new ApiError(res.status, `${res.status}`);
  return (await res.json()) as T;
}

export const api = {
  me: () => getJSON<Me>("/v1/me"),

  voiceprints: () =>
    getJSON<{ email: string; count: number; voiceprints: Voiceprint[] }>(
      "/v1/me/voiceprints",
    ),

  setFlags: (
    workspaceId: string,
    voiceprintId: string,
    flags: { identify_allowed?: boolean; enroll_allowed?: boolean },
  ) =>
    postJSON<{ identify_allowed: boolean; enroll_allowed: boolean }>(
      `/v1/me/voiceprints/${encodeURIComponent(workspaceId)}/${encodeURIComponent(voiceprintId)}/flags`,
      flags,
    ),

  forget: (workspaceId: string, voiceprintId: string) =>
    postJSON<{ deleted: boolean }>(
      `/v1/me/voiceprints/${encodeURIComponent(workspaceId)}/${encodeURIComponent(voiceprintId)}/forget`,
    ),

  // P4 consent inbox: proposals where someone tagged this user's voice.
  pending: () => getJSON<{ email: string; pending: Proposal[] }>("/v1/me/pending"),
  confirm: (proposalId: string) =>
    postJSON<{ status: string }>("/v1/confirm", { proposal_id: proposalId }),
  deny: (proposalId: string) =>
    postJSON<{ status: string }>("/v1/deny", { proposal_id: proposalId }),

  logout: () => postJSON<{ ok: boolean }>("/auth/logout"),
};

/** Full-page navigations (these set/clear the cookie on the FastAPI side). */
export const authUrls = {
  google: "/auth/login",
  devLogin: (email: string) => `/auth/dev-login?email=${encodeURIComponent(email)}`,
};
