import { NextRequest, NextResponse } from "next/server";
import { clerkClient, auth } from "@clerk/nextjs/server";

// The Haki API exposes no CORS headers, so the console never calls it
// directly from the browser: this route forwards requests server-side.
//
// Two auth modes, same as lib/api.ts:
// - self-hosted/raw key: the client sends its own Authorization header,
//   forwarded as-is — key never touches this server beyond the request.
// - Clerk-provisioned: the client sends NO Authorization header; if a
//   Clerk session exists, the real hk_ key is read here from that user's
//   private metadata (server-only, set by app/api/provision/route.ts) and
//   injected — the browser never sees it.
//
// Security: this proxy is reach-able by ANY browser tab, so it must not be
// an open relay. Only the console's own API surface is forwarded: a request
// outside the allowlist gets 404 before any fetch, so neither the service
// key injected below nor upstream error text can ever reach attacker-chosen
// endpoints (orgs/provision, billing, webhooks, admin routes).
const API_URL = (
  process.env.NEXT_PUBLIC_HAKI_API_URL ?? "http://localhost:8100"
).replace(/\/$/, "");

const ALLOWED_PREFIXES = [
  "v1/facts",
  "v1/events",
  "v1/timeline",
  "v1/context",
  "v1/graph",
  "v1/traces",
  "v1/keys",
  "v1/orgs/settings",
  "v1/orgs/members",
  "v1/insights",
  "v1/usage",
  "v1/projects",
  "v1/capture",
  "v1/forget",
];

function isAllowed(path: string[]): boolean {
  const joined = path.join("/");
  // Reject any path segment that decodes to traversal characters — Next
  // already decodes catch-all params, so a literal ".." here means someone
  // crafted the URL to escape the allowlist prefix.
  if (path.some((seg) => seg === "." || seg === "..")) return false;
  return ALLOWED_PREFIXES.some(
    (p) => joined === p || joined.startsWith(`${p}/`),
  );
}

async function handler(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
): Promise<NextResponse> {
  const { path } = await context.params;
  if (!isAllowed(path)) {
    return NextResponse.json(
      { error: { type: "not_found", message: "Unknown API route." } },
      { status: 404 },
    );
  }
  const target = `${API_URL}/${path.join("/")}${request.nextUrl.search}`;

  const headers: Record<string, string> = {};
  const authorization = request.headers.get("authorization");
  if (authorization) {
    headers.Authorization = authorization;
  } else {
    const { userId } = await auth();
    if (userId) {
      const client = await clerkClient();
      const user = await client.users.getUser(userId);
      const apiKey = (user.privateMetadata as { hakiApiKey?: string }).hakiApiKey;
      if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
    }
  }
  const contentType = request.headers.get("content-type");
  if (contentType) headers["Content-Type"] = contentType;

  const init: RequestInit = { method: request.method, headers };
  if (request.method !== "GET" && request.method !== "HEAD") {
    init.body = await request.text();
  }

  try {
    const upstream = await fetch(target, init);
    const body = await upstream.text();
    return new NextResponse(body, {
      status: upstream.status,
      headers: {
        "Content-Type":
          upstream.headers.get("content-type") ?? "application/json",
      },
    });
  } catch {
    // No URL interpolation here: the console is public, the message would
    // leak the internal API address to any visitor.
    return NextResponse.json(
      {
        error: {
          type: "api_unreachable",
          message:
            "The Haki API is unreachable. Check its configuration or try again later.",
        },
      },
      { status: 502 },
    );
  }
}

export {
  handler as GET,
  handler as POST,
  handler as PUT,
  handler as PATCH,
  handler as DELETE,
};
