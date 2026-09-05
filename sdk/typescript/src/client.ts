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
  /**
   * Packet-local reference the rendered block cites (`F3`), and the exact
   * line to print for this item. Both server-rendered since 22 aout: the
   * block used to be built independently here, in the Python SDK and in the
   * MCP server, while the token budget was computed from a fourth string
   * matching none of them. Absent when talking to an older server, in which
   * case buildPromptContext falls back to rendering from the fields below.
   */
  ref?: string | null;
  line?: string | null;
  predicate: string;
  value: Record<string, unknown>;
  confidence: number | null;
  valid_from: string | null;
  /** valid_from without the seconds and UTC offset nothing reads. */
  valid_from_short?: string | null;
  /** Dual-date rendering (mechanism F1, 15 aout): exact offset from the
   * temporal point of view ("N days before/after the question"),
   * precomputed server-side. Absent on older servers. */
  valid_from_relative?: string | null;
  /** {"start": iso, "end": iso} when this fact's source text used a
   * relative time expression the extractor resolved. Absent otherwise or
   * on older servers. */
  temporal_range?: { start: string; end: string } | null;
  /** Reclassification safety net (16 aout): true when this fact was
   * activated by the automatic overflow reclassification (mechanism C)
   * rather than an extractor declaring memory_form="event" up front.
   * Absent/false on older servers. */
  auto_reclassified?: boolean;
  source_event_ids: string[];
  /** Typology + volatility (M2); absent when talking to a pre-M2 server. */
  fact_kind?: string | null;
  volatility?: string | null;
  last_confirmed?: string | null;
  freshness?: "current" | "unconfirmed" | "stale" | null;
  /** What authority this fact was born with (M8). Absent on older servers. */
  origin_trust?: "trusted" | "semi_trusted" | "third_party" | "untrusted";
  /** Who actually said it, when a third party did. */
  attributed_to?: string | null;
  /** Open conflicts (13 aout): true when served alongside a genuinely
   * conflicting sibling (same `conflict_id`) instead of being hidden.
   * Absent on older servers. */
  contested?: boolean;
  conflict_id?: string | null;
}

/** Source event excerpt served in the packet (episodic memory). */
export interface PacketEpisode {
  event_id: string;
  /** See PacketFact.ref / PacketFact.line. */
  ref?: string | null;
  line?: string | null;
  kind: string;
  occurred_at: string | null;
  /** occurred_at without the seconds and UTC offset nothing reads. */
  occurred_at_short?: string | null;
  /** Dual-date rendering (mechanism F1, 15 aout) -- see
   * PacketFact.valid_from_relative. Absent on older servers. */
  occurred_at_relative?: string | null;
  excerpt: string;
  /** Context window (mechanism F2, 15 aout): true when this episode was
   * added as the temporal neighbor of a score-packed episode, or as the
   * source turn of a packed fact -- not itself a scored/ranked inclusion. */
  context_neighbor?: boolean;
}

/** Noisy-failure contract, parity with app.schemas.context.ContextStatus:
 * "ok" = nothing to report, "degraded" = a packet was produced but something
 * is worth flagging (see `warnings`), "failed" = no real packet could be
 * built (see the Python SDK's app.context.failed_packet). */
export type ContextStatus = "ok" | "degraded" | "failed";

export interface ContextPacket {
  /** What the rendered block costs BESIDES the items (22 aout): the fixed
   * instruction paragraphs and the delimiters. Not part of budget_tokens.
   * Absent on older servers. */
  overhead_tokens?: number;
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
  feedback_deleted: number;
  predicate_aliases_deleted: number;
  subject_aliases_deleted: number;
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

// Statuses worth a second attempt: rate limiting and transient upstream
// failures — same contract as the Python SDK (sdk/python/src/haki/client.py).
// Anything else (400/401/403/404/409/422, ...) is definitive.
const RETRYABLE_STATUS = new Set([429, 500, 502, 503, 504]);

// Extra attempts after the first one: 3 retries = 4 tries total, backoff
// 0.5 s / 1 s / 2 s plus jitter so a fleet of agents does not retry in
// lockstep.
const MAX_RETRIES = 3;
const BACKOFF_BASE_MS = 500;

function retryDelayMs(attempt: number, retryAfter: string | null): number {
  if (retryAfter !== null) {
    const asked = Number(retryAfter);
    // The server tells us how long to wait — believe it, within reason.
    if (Number.isFinite(asked) && asked >= 0 && asked <= 60) {
      return asked * 1000;
    }
  }
  return BACKOFF_BASE_MS * 2 ** (attempt - 1) + Math.random() * 250;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

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
    retry = true,
  ): Promise<any> {
    const url = new URL(this.baseUrl + path);
    for (const [name, value] of Object.entries(options.params ?? {})) {
      url.searchParams.set(name, value);
    }
    const headers: Record<string, string> = {};
    if (this.apiKey) headers["Authorization"] = `Bearer ${this.apiKey}`;
    if (options.body !== undefined) headers["Content-Type"] = "application/json";

    // Retries 429/500/502/503/504 and network errors with exponential
    // backoff (+ Retry-After on 429). Pass `retry=false` for calls that
    // must run exactly once — today only `createKey`, where a retry after
    // an ambiguous timeout would mint a second key and orphan the first.
    let lastResponse: Response | undefined;
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      let response: Response;
      try {
        response = await fetch(url, {
          method,
          headers,
          body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
          signal: AbortSignal.timeout(this.timeout),
        });
      } catch (err) {
        if (!retry || attempt >= MAX_RETRIES) {
          const detail = err instanceof Error ? err.message : String(err);
          throw new HakiConnectionError(
            `cannot reach Haki at ${this.baseUrl}: ${detail}`,
          );
        }
        await sleep(retryDelayMs(attempt + 1, null));
        continue;
      }
      if (response.ok || !RETRYABLE_STATUS.has(response.status)) {
        await HakiClient.throwForStatus(response);
        return response.json();
      }
      lastResponse = response;
      if (!retry || attempt >= MAX_RETRIES) {
        break;
      }
      await sleep(retryDelayMs(attempt + 1, response.headers.get("retry-after")));
    }
    // Exhausted on a retryable status: raise the TYPED error from the last
    // body, same as a first-try failure — never a bare status.
    await HakiClient.throwForStatus(lastResponse!);
    return lastResponse!.json();
  }

  /** Raise HakiApiError for a non-2xx response, parsing the typed body. */
  private static async throwForStatus(response: Response): Promise<void> {
    if (response.ok) {
      return;
    }
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

  /**
   * Assemble a ContextPacket. Returns {packet, token_count, trace_id}.
   *
   * `excludeIds` asks for the NEXT PAGE of the same ranked list: pass what
   * an earlier packet already gave you (see `seen` in runtime.ts) and those
   * items are dropped before ranking. Ask again with the SAME query --
   * rewriting it with what you just read is measurably worse. This is a
   * further page, not a second hop.
   */
  context(options: {
    subjectId: string;
    query: string;
    projectId: string;
    purpose?: string;
    budgetTokens?: number;
    excludeIds?: string[];
  }): Promise<ContextResponse> {
    return this.request("POST", "/v1/context", {
      body: {
        project_id: options.projectId,
        subject_id: options.subjectId,
        query: options.query,
        purpose: options.purpose ?? null,
        budget_tokens: options.budgetTokens ?? 3000,
        exclude_ids: options.excludeIds ?? null,
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

  /**
   * Process pending/failed consolidation jobs now. Returns {processed}.
   *
   * Dev/ops endpoint: it drains the pending jobs of EVERY project on the
   * server, on a session without RLS scoping, and requires the admin key.
   * Fine on a local dev server, wrong against a shared one -- prefer
   * `consolidateSubject` whenever the subject is known.
   */
  consolidate(): Promise<ConsolidateResponse> {
    return this.request("POST", "/v1/consolidate");
  }

  /**
   * Consolidate one subject's pending jobs now. Returns {processed}.
   *
   * The scoped counterpart of `consolidate()`: same synchronous
   * "extraction happened, look now" behavior, but bounded to one
   * project/subject, so a caller never triggers work on another tenant's
   * data, never waits behind it, and needs only a normal project key.
   */
  consolidateSubject(options: {
    projectId: string;
    subjectId: string;
  }): Promise<ConsolidateResponse> {
    return this.request("POST", "/v1/consolidate/subject", {
      params: { project_id: options.projectId, subject_id: options.subjectId },
    });
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
    return this.request(
      "POST",
      "/v1/keys",
      {
        body: {
          org_id: options.orgId,
          project_id: options.projectId,
          label: options.label ?? null,
        },
      },
      false, // never replay: a retry could mint an orphan key
    );
  }

  /** Masked key listing (prefix only). */
  listKeys(): Promise<KeyListResponse> {
    return this.request("GET", "/v1/keys");
  }

  revokeKey(keyId: string): Promise<KeyRevokedResponse> {
    return this.request("DELETE", `/v1/keys/${keyId}`);
  }
}
