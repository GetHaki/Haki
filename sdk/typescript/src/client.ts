/** Haki TypeScript SDK client (fetch natif Node 18+, zéro dépendance runtime).
 *
 * Wraps the Haki HTTP API with readable, typed errors — parity with the
 * Python SDK (sdk/python/src/haki/client.py). All methods are async; the
 * wire format (snake_case) is kept in the exported interfaces.
 */

import { HakiApiError, HakiConnectionError } from "./errors.js";

// -- Wire types (snake_case, contract B.1 and route schemas) ------------------

/** Source event, contract B.1. */
export interface EventInput {
  org_id: string;
  project_id: string;
  subject_type?: string;
  subject_id?: string;
  actor_type?: string;
  actor_id?: string;
  agent_id?: string;
  thread_id?: string;
  run_id?: string;
  /** Origin trust (M8): trusted | semi_trusted | third_party | untrusted.
   * Omitted -> the server derives it from actor_type. */
  origin_trust?: string;
  kind: string;
  occurred_at: string; // ISO 8601
  payload: Record<string, unknown>;
  source?: Record<string, unknown>;
  classification?: string[];
  retention_policy?: string;
  idempotency_key?: string;
}

export interface CaptureResponse {
  status: string;
  events: { id: string; deduplicated: boolean }[];
  /** null when the whole batch was deduplicated. */
  consolidation_job_id: string | null;
  policy: string;
}

export interface PacketFact {
  id: string;
  predicate: string;
  value: Record<string, unknown>;
  confidence: number | null;
  valid_from: string | null;
  source_event_ids: string[];
  /** Typology + volatility (M2); absent when talking to a pre-M2 server. */
  fact_kind?: string | null;
  volatility?: string | null;
  last_confirmed?: string | null;
  freshness?: "current" | "unconfirmed" | null;
  /** What authority this fact was born with (M8). Absent on older servers. */
  origin_trust?: "trusted" | "semi_trusted" | "third_party" | "untrusted";
  /** Who actually said it, when a third party did. */
  attributed_to?: string | null;
}

/** Source event excerpt served in the packet (episodic memory). */
export interface PacketEpisode {
  event_id: string;
  kind: string;
  occurred_at: string | null;
  excerpt: string;
}

/** Noisy-failure contract, parity with app.schemas.context.ContextStatus:
 * "ok" = nothing to report, "degraded" = a packet was produced but something
 * is worth flagging (see `warnings`), "failed" = no real packet could be
 * built (see the Python SDK's app.context.failed_packet). */
export type ContextStatus = "ok" | "degraded" | "failed";

export interface ContextPacket {
  facts: PacketFact[];
  episodes: PacketEpisode[];
  /** Doubles as the typed list of reasons for `status`. */
  warnings: string[];
  status: ContextStatus;
  /** M3 recall gate: "no_relevant_memory" when the relevance floor emptied
   * the packet although the subject has memories. Absent/null otherwise —
   * deliberately not a warning (status stays "ok"). */
  empty_reason?: "no_relevant_memory" | null;
}

export interface ContextResponse {
  packet: ContextPacket;
  token_count: number;
  trace_id: string;
}

export interface TraceResponse {
  trace_id: string;
  project_id: string;
  subject_id: string;
  query: string;
  purpose: string | null;
  packet: ContextPacket;
  decisions: { fact_id: string; action: string; reason_code: string }[];
  token_count: number;
}

export interface EventOut {
  id: string;
  org_id: string;
  project_id: string;
  subject_type: string;
  subject_id: string;
  actor_type: string | null;
  actor_id: string | null;
  agent_id: string | null;
  thread_id: string | null;
  run_id: string | null;
  kind: string;
  occurred_at: string;
  recorded_at: string;
  payload: Record<string, unknown>;
  source: Record<string, unknown> | null;
  classification: string[];
  retention_policy: string | null;
  hash: string;
  idempotency_key: string;
}

export interface TimelineResponse {
  events: EventOut[];
}

export interface ConsolidateResponse {
  processed: number;
}

export type ForgetMode = "disable" | "delete";

export interface ForgetResponse {
  status: string;
  mode: ForgetMode;
  scope: "fact" | "subject";
  forget_id: string;
  facts_disabled: number;
  facts_deleted: number;
  conflict_sets_deleted: number;
  events_deleted: number;
  traces_deleted: number;
}

export type FeedbackRating = "useful" | "irrelevant" | "incorrect";

export interface FeedbackResponse {
  status: string;
  feedback_id: string;
  /** Resulting fact status when the feedback targeted a fact, else null. */
  fact_status: string | null;
}

export interface ResolveConflictResponse {
  conflict_id: string;
  status: string;
  kept_fact_id: string;
  superseded_fact_ids: string[];
  resolved_at: string;
}

export interface KeyCreatedResponse {
  id: string;
  /** The clear key, returned ONLY here at creation. Store it. */
  key: string;
  prefix: string;
  org_id: string;
  project_id: string;
  label: string | null;
  created_at: string;
}

export interface KeyOut {
  id: string;
  prefix: string;
  org_id: string;
  project_id: string;
  label: string | null;
  created_at: string;
  revoked_at: string | null;
}

export interface KeyListResponse {
  keys: KeyOut[];
}

export interface KeyRevokedResponse {
  id: string;
  status: string;
}

export interface HealthResponse {
  status: string;
  database: string;
}

// -- Client --------------------------------------------------------------------

export interface HakiClientOptions {
  /** Base URL of the Haki API, e.g. "http://localhost:8100". */
  baseUrl: string;
  /** API key (hk_...). Sent as `Authorization: Bearer` when set. */
  apiKey?: string;
  /** Request timeout in milliseconds (default 10_000). */
  timeout?: number;
}

export class HakiClient {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly timeout: number;

  constructor(options: HakiClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.timeout = options.timeout ?? 10_000;
  }

  private async request(
    method: string,
    path: string,
    options: { params?: Record<string, string>; body?: unknown } = {},
  ): Promise<any> {
    const url = new URL(this.baseUrl + path);
    for (const [name, value] of Object.entries(options.params ?? {})) {
      url.searchParams.set(name, value);
    }
    const headers: Record<string, string> = {};
    if (this.apiKey) headers["Authorization"] = `Bearer ${this.apiKey}`;
    if (options.body !== undefined) headers["Content-Type"] = "application/json";

    let response: Response;
    try {
      response = await fetch(url, {
        method,
        headers,
        body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
        signal: AbortSignal.timeout(this.timeout),
      });
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      throw new HakiConnectionError(
        `cannot reach Haki at ${this.baseUrl}: ${detail}`,
      );
    }

    if (!response.ok) {
      let errorType: string | undefined;
      let message: string | undefined;
      let field: string | undefined;
      let payload: Record<string, unknown> = {};
      try {
        payload = (await response.json()) as Record<string, unknown>;
        const error = (payload.error ?? {}) as Record<string, unknown>;
        errorType = error.type as string | undefined;
        message = error.message as string | undefined;
        field = error.field as string | undefined;
      } catch {
        // Non-JSON error body: fall back to the HTTP status only.
      }
      const text = message ?? `HTTP ${response.status}`;
      throw new HakiApiError(text, {
        statusCode: response.status,
        errorType,
        field,
        payload,
      });
    }
    return response.json();
  }

  // -- API (parity with the Python HakiClient) --------------------------------

  health(): Promise<HealthResponse> {
    return this.request("GET", "/health");
  }

  /** Ingest events (contract B.1). Idempotent per idempotencyKey. */
  capture(
    events: EventInput[],
    idempotencyKey?: string,
  ): Promise<CaptureResponse> {
    return this.request("POST", "/v1/capture", {
      body: { events, idempotency_key: idempotencyKey ?? null },
    });
  }

  /** Assemble a ContextPacket. Returns {packet, token_count, trace_id}. */
  context(options: {
    subjectId: string;
    query: string;
    projectId: string;
    purpose?: string;
    budgetTokens?: number;
  }): Promise<ContextResponse> {
    return this.request("POST", "/v1/context", {
      body: {
        project_id: options.projectId,
        subject_id: options.subjectId,
        query: options.query,
        purpose: options.purpose ?? null,
        budget_tokens: options.budgetTokens ?? 900,
      },
    });
  }

  /** Full decision trace of a context call (scope mandatory). */
  inspect(
    traceId: string,
    scope: { projectId: string; subjectId: string },
  ): Promise<TraceResponse> {
    return this.request("GET", `/v1/inspect/${traceId}`, {
      params: { project_id: scope.projectId, subject_id: scope.subjectId },
    });
  }

  /** Events of one subject, ordered by occurred_at. */
  timeline(scope: {
    projectId: string;
    subjectId: string;
  }): Promise<TimelineResponse> {
    return this.request("GET", "/v1/timeline", {
      params: { project_id: scope.projectId, subject_id: scope.subjectId },
    });
  }

  /** Process pending/failed consolidation jobs now. Returns {processed}. */
  consolidate(): Promise<ConsolidateResponse> {
    return this.request("POST", "/v1/consolidate");
  }

  /** Forget one fact or one subject (exactly one target required). */
  forget(options: {
    projectId: string;
    mode?: ForgetMode;
    subjectId?: string;
    factId?: string;
  }): Promise<ForgetResponse> {
    return this.request("POST", "/v1/forget", {
      body: {
        project_id: options.projectId,
        subject_id: options.subjectId ?? null,
        fact_id: options.factId ?? null,
        mode: options.mode ?? "disable",
      },
    });
  }

  /** Quality observation on a trace or a fact (exactly one target). */
  feedback(options: {
    projectId: string;
    rating: FeedbackRating;
    traceId?: string;
    factId?: string;
    comment?: string;
  }): Promise<FeedbackResponse> {
    return this.request("POST", "/v1/feedback", {
      body: {
        project_id: options.projectId,
        trace_id: options.traceId ?? null,
        fact_id: options.factId ?? null,
        rating: options.rating,
        comment: options.comment ?? null,
      },
    });
  }

  /** Resolve an open conflict set: keep one fact, supersede the others. */
  resolveConflict(
    conflictId: string,
    options: { projectId: string; keepFactId: string },
  ): Promise<ResolveConflictResponse> {
    return this.request("POST", `/v1/conflicts/${conflictId}/resolve`, {
      body: { project_id: options.projectId, keep_fact_id: options.keepFactId },
    });
  }

  /** Create an API key. The clear key is in the response ONCE — store it. */
  createKey(options: {
    projectId: string;
    orgId: string;
    label?: string;
  }): Promise<KeyCreatedResponse> {
    return this.request("POST", "/v1/keys", {
      body: {
        org_id: options.orgId,
        project_id: options.projectId,
        label: options.label ?? null,
      },
    });
  }

  /** Masked key listing (prefix only). */
  listKeys(): Promise<KeyListResponse> {
    return this.request("GET", "/v1/keys");
  }

  revokeKey(keyId: string): Promise<KeyRevokedResponse> {
    return this.request("DELETE", `/v1/keys/${keyId}`);
  }
}
