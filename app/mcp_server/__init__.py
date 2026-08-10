"""Haki MCP server (sprint 4) — Cursor integration.

Five tools: haki_context, haki_capture, haki_inspect, haki_forget (the
original PRD four), plus haki_correct (M10 — corrects the memory from
inside the conversation itself, the MCP-side entry point to the same
mechanism as POST /v1/feedback). Transport: Streamable HTTP, mounted
inside the existing FastAPI app on /mcp (one single server to run).

Noisy-failure contract (extends X-Haki-Memory — app/gateway/__init__.py —
to this surface): haki_context and haki_inspect always return a `status`
("ok" | "degraded" | "failed", see app.schemas.context.ContextStatus)
alongside the packet, and `warnings` doubles as the typed reasons. A
build_context failure inside haki_context is caught, logged loudly
(logger.exception — never a silent pass), counted (app.metrics), and
turned into a status="failed" result instead of either a raw MCP protocol
error or, worse, a result that looks like "the subject has no memory".

Memory is always PROJECT- AND SUBJECT-SCOPED, never chosen by the model
(PRD — "Le client ne doit pas laisser le modele choisir ces valeurs").
Neither tool accepts a subject_id argument — a team sharing one Cursor
deployment needs one config per person, the same way each install already
gets its own project. Two scoping sources, resolved per-request by
_resolve_scope below (the original single-server-config design only works
for a self-hosted `docker compose up` install — one process for one
person; a server shared by several tenants from ONE process can't hold
"this caller's project" in a global env var):

- A real `hk_` API key (Authorization: Bearer hk_..., the same key used
  against /v1/*) resolves org_id/project_id exactly like
  ApiKeyAuthMiddleware does for /v1/* (app/auth.py); subject_id then
  comes from a client-set X-Haki-Subject-Id header (configured once in
  the MCP client's own config, not per-call).
- No Authorization header at all falls back to the legacy single-server
  config (HAKI_MCP_PROJECT_ID/ORG_ID/SUBJECT_ID) — unchanged behavior for
  existing self-hosted installs.

Auth: a request bearing a real `hk_` key is validated per-tool-call
against the api_keys table (revoked/unknown -> ToolError, never silently
falls through to the legacy path). A request with no Authorization header
at all still respects the legacy single-shared-secret gate (HAKI_API_KEY,
enforced in app.main) for self-hosted dev use. Full OAuth: later sprint.

Honest limit (PRD / Haki_Memory_Runtime.md): MCP alone does NOT intercept
100 % of Cursor conversations — the server only sees the tool calls Cursor
decides to make. The Project Rule installed by `haki mcp` instructs the
agent when to call these tools; coverage is measured, never promised.
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError

from app import ledger, metrics
from app.auth import resolve_api_key
from app.config import settings
from app.context import build_context, failed_packet, get_trace
from app.db import async_session
from app.schemas import EventIn

logger = logging.getLogger("haki.mcp")

mcp = MCPServer(
    "haki",
    instructions=(
        "Memoire long-terme du projet. Appeler haki_context avant de planifier "
        "ou modifier du code, haki_capture en fin de tache pour les decisions, "
        "conventions et erreurs resolues. Ne memoriser que du durable, jamais "
        "de secrets. Appeler haki_correct quand l'utilisateur signale qu'un "
        "fait rappele par haki_context est errone."
    ),
)


def _format_packet(subject_id: str, packet: dict[str, Any]) -> str:
    """Render a ContextPacket as a text block ready to inject in a prompt."""
    lines = [f"# Memoire projet Haki (sujet {subject_id})"]
    status = packet.get("status", "ok")
    if status != "ok":
        lines.append(
            f"[memoire {status}] la memoire peut etre incomplete pour cet appel."
        )
    facts = packet.get("facts", [])
    if not facts:
        lines.append("Aucun fait memorise pour ce sujet dans ce projet.")
    for fact in facts:
        value = json.dumps(fact["value"], ensure_ascii=False, sort_keys=True)
        when = fact.get("valid_from") or "date inconnue"
        sources = ", ".join(fact.get("source_event_ids") or []) or "aucune"
        suffix = ""
        if fact.get("freshness") == "unconfirmed":
            last = fact.get("last_confirmed") or "date inconnue"
            suffix = f" [A RECONFIRMER — derniere confirmation {last}]"
        lines.append(
            f"- {fact['predicate']} = {value} (valide depuis {when}; "
            f"source: {sources}){suffix}"
        )
    for warning in packet.get("warnings", []):
        lines.append(f"Attention: {warning}")
    return "\n".join(lines)


class _Scope:
    __slots__ = ("org_id", "project_id", "subject_id")

    def __init__(self, org_id: str, project_id: str, subject_id: str) -> None:
        self.org_id = org_id
        self.project_id = project_id
        self.subject_id = subject_id


async def _resolve_scope(ctx: Context) -> _Scope:
    """Real multi-tenant scoping for a shared /mcp endpoint.

    Every tool below used to read project_id/org_id/subject_id straight
    off HAKI_MCP_* server config — fine for a `docker compose up` install
    where the whole server exists for exactly one person, broken for a
    server shared by several tenants from ONE process (a single global
    env var cannot hold "this caller's project"). A caller's own `hk_`
    key — the same one already used for the REST API — now resolves
    org_id/project_id here exactly like ApiKeyAuthMiddleware does for
    /v1/* (app/auth.py). subject_id has no equivalent server-side source
    (MCP tools take no subject_id argument by design — PRD: "le modele ne
    choisit pas ce scope"), so it travels as a client-set HTTP header,
    X-Haki-Subject-Id, configured once in the MCP client's own config
    (mcp.json), not chosen per-call by the model.

    No Authorization header at all -> unchanged self-hosted behavior
    (HAKI_MCP_PROJECT_ID/ORG_ID/SUBJECT_ID), so `docker compose up` and
    existing self-hosted installs keep working exactly as before.
    """
    headers = ctx.headers or {}
    auth_header = headers.get("authorization") or headers.get("Authorization")
    token = None
    if auth_header and auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()

    if token is None:
        return _Scope(settings.mcp_org_id, settings.mcp_project_id, settings.mcp_subject_id)

    if not token.startswith("hk_"):
        raise ToolError(
            "invalid Authorization header: expected 'Bearer hk_...' (a Haki API key), "
            "not the legacy single shared secret"
        )
    key = await resolve_api_key(token)
    if key is None:
        raise ToolError("invalid or revoked API key")

    subject_id = headers.get("x-haki-subject-id") or headers.get("X-Haki-Subject-Id")
    if not subject_id:
        raise ToolError(
            "missing X-Haki-Subject-Id header: set it once in your MCP client config "
            "(mcp.json) to the subject this server instance should remember — a team "
            "sharing one Cursor deployment needs one subject per person, the same way "
            "each install already gets its own project"
        )
    return _Scope(key.org_id, key.project_id, subject_id)


@mcp.tool()
async def haki_context(ctx: Context, query: str, budget_tokens: int = 900) -> dict[str, Any]:
    """Rappelle la memoire du projet (decisions, conventions, preferences)
    pertinente pour une tache. A appeler AVANT de planifier ou modifier du
    code. Retourne un bloc pret a injecter, avec dates et sources, le
    trace_id (inspectable via haki_inspect), et un `status` explicite
    ("ok"/"degraded"/"failed") — ne jamais traiter une reponse degradee ou
    en echec comme "aucun fait connu"."""
    scope = await _resolve_scope(ctx)
    try:
        async with async_session() as session:
            packet, token_count, trace_id = await build_context(
                session,
                project_id=scope.project_id,
                subject_id=scope.subject_id,
                query=query,
                purpose="mcp",
                budget_tokens=budget_tokens,
            )
            await session.commit()
    except Exception:
        # Loud, not silent: logged, counted, and reported back to the agent
        # as an explicit status="failed" result — same spirit as the
        # gateway's X-Haki-Memory: degraded (app/api/routes/gateway.py),
        # applied to the MCP surface. The tool call still succeeds (no raw
        # MCP protocol error): a caller that only reads `facts` would
        # otherwise be unable to tell "no memory" from "memory unreachable".
        logger.exception(
            "mcp haki_context: build_context failed (project=%s subject=%s)",
            scope.project_id,
            scope.subject_id,
        )
        metrics.increment("mcp.context.failed")
        packet = failed_packet(["build_context_failed: see server logs"])
        token_count = 0
        trace_id = None
    else:
        metrics.increment(f"mcp.context.{packet['status']}")
    return {
        "context": _format_packet(scope.subject_id, packet),
        "facts": packet["facts"],
        "warnings": packet["warnings"],
        "status": packet["status"],
        "token_count": token_count,
        "trace_id": str(trace_id) if trace_id else None,
        "project_id": scope.project_id,
    }


@mcp.tool()
async def haki_capture(ctx: Context, content: str, kind: str = "agent.observation") -> dict[str, Any]:
    """Memorise un fait durable du projet : decision technique, convention,
    erreur resolue. A appeler en FIN de tache. Ne jamais memoriser de
    secrets, tokens ou donnees ephemeres. La consolidation est synchrone en
    dev (HAKI_MCP_AUTOCONSOLIDATE), donc le fait est rappelable
    immediatement."""
    scope = await _resolve_scope(ctx)
    subject_id = scope.subject_id
    # Content-based idempotency key (no timestamp): an agent calling the
    # tool twice with the same memory does not create a duplicate event.
    digest = hashlib.sha256(f"{kind}\n{content}".encode("utf-8")).hexdigest()
    event = EventIn(
        org_id=scope.org_id,
        project_id=scope.project_id,
        subject_type="user",
        subject_id=subject_id,
        actor_type="agent",
        agent_id="cursor",
        kind=kind,
        occurred_at=datetime.now(timezone.utc),
        payload={"role": "assistant", "content": content},
        source={"tool": "mcp", "server": "haki"},
        idempotency_key=f"mcp:{subject_id}:{digest}",
    )
    async with async_session() as session:
        results = await ledger.write_events(session, [event])
        stored, deduplicated = results[0]
        job = None
        if not deduplicated:
            job = await ledger.create_consolidation_job(
                session,
                project_id=scope.project_id,
                event_ids=[stored.id],
            )
        await session.commit()

        processed = None
        if settings.mcp_autoconsolidate and job is not None:
            processed = await ledger.run_pending_consolidations(session)
            await session.commit()

    return {
        "event_id": str(stored.id),
        "deduplicated": deduplicated,
        "consolidated": processed,
        "project_id": scope.project_id,
    }


@mcp.tool()
async def haki_inspect(ctx: Context, trace_id: str) -> dict[str, Any]:
    """Inspecte la trace d'un appel haki_context : quels faits ont ete
    inclus, exclus ou bloques, et pourquoi (reason_code). Preuve de
    provenance de la memoire servie."""
    scope = await _resolve_scope(ctx)
    async with async_session() as session:
        trace = await get_trace(
            session,
            trace_id=uuid.UUID(trace_id),
            project_id=scope.project_id,
            subject_id=scope.subject_id,
        )
    return {
        "trace_id": str(trace.id),
        "project_id": trace.project_id,
        "subject_id": trace.subject_id,
        "query": trace.query,
        "purpose": trace.purpose,
        "token_count": trace.token_count,
        "packet": trace.packet,
        "status": (trace.packet or {}).get("status", "ok"),
        "decisions": trace.decisions,
    }


@mcp.tool()
async def haki_forget(ctx: Context, mode: str = "disable") -> dict[str, Any]:
    """Oublie la memoire du sujet configure pour ce serveur, dans ce projet.
    mode='disable' (reversible, les faits passent a disabled) ou 'delete'
    (effacement reel : faits, embeddings, evenements, traces). Retourne le
    recu d'effacement (forget_id) et les compteurs de ce qui a ete fait."""
    scope = await _resolve_scope(ctx)
    async with async_session() as session:
        receipt, counters = await ledger.forget(
            session,
            project_id=scope.project_id,
            subject_id=scope.subject_id,
            mode=mode,
        )
        await session.commit()
    return {
        "status": "ok",
        "mode": mode,
        "scope": receipt.scope,
        "forget_id": str(receipt.id),
        "project_id": scope.project_id,
        **counters,
    }


@mcp.tool()
async def haki_correct(
    ctx: Context,
    rating: str,
    fact_id: str | None = None,
    trace_id: str | None = None,
    comment: str | None = None,
) -> dict[str, Any]:
    """Corrige la memoire depuis la conversation (M10). rating='incorrect'
    avec un fact_id fait passer ce fait a 'disputed' : haki_context ne le
    rappellera plus jamais. rating='useful'/'irrelevant' journalise un avis
    sur un rappel sans changer le fait. Exactement une cible requise :
    fact_id (un fait precis, generalement vu via haki_context/haki_inspect)
    OU trace_id (un appel haki_context entier). Meme mecanisme que
    POST /v1/feedback (app.ledger.submit_feedback) : effet identique quel
    que soit le chemin d'appel. Scope resolu par cle API (voir
    _resolve_scope) ou, en self-hosted sans cle, par HAKI_MCP_PROJECT_ID."""
    scope = await _resolve_scope(ctx)
    async with async_session() as session:
        row, fact_status = await ledger.submit_feedback(
            session,
            project_id=scope.project_id,
            rating=rating,
            trace_id=uuid.UUID(trace_id) if trace_id else None,
            fact_id=uuid.UUID(fact_id) if fact_id else None,
            comment=comment,
        )
        await session.commit()
    return {
        "status": "recorded",
        "feedback_id": str(row.id),
        "fact_status": fact_status,
        "project_id": scope.project_id,
    }
