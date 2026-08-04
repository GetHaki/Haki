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

Memory is PROJECT- AND SUBJECT-SCOPED by config: project_id and subject_id
both come from the server config (HAKI_MCP_PROJECT_ID, HAKI_MCP_SUBJECT_ID),
never from the model (PRD — "Le client ne doit pas laisser le modele
choisir ces valeurs"). Neither tool accepts a subject_id argument — a team
sharing one Cursor deployment needs one server config per person, the same
way each install already gets its own project.

Dev auth: when HAKI_API_KEY is set, the /mcp endpoint requires
`Authorization: Bearer <key>` (enforced in app.main). Unset = open mode,
documented for local development only. Full OAuth: later sprint.

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

from app import ledger, metrics
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
        lines.append(
            f"- {fact['predicate']} = {value} (valide depuis {when}; "
            f"source: {sources})"
        )
    for warning in packet.get("warnings", []):
        lines.append(f"Attention: {warning}")
    return "\n".join(lines)


@mcp.tool()
async def haki_context(query: str, budget_tokens: int = 900) -> dict[str, Any]:
    """Rappelle la memoire du projet (decisions, conventions, preferences)
    pertinente pour une tache. A appeler AVANT de planifier ou modifier du
    code. Retourne un bloc pret a injecter, avec dates et sources, le
    trace_id (inspectable via haki_inspect), et un `status` explicite
    ("ok"/"degraded"/"failed") — ne jamais traiter une reponse degradee ou
    en echec comme "aucun fait connu"."""
    try:
        async with async_session() as session:
            packet, token_count, trace_id = await build_context(
                session,
                project_id=settings.mcp_project_id,
                subject_id=settings.mcp_subject_id,
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
            settings.mcp_project_id,
            settings.mcp_subject_id,
        )
        metrics.increment("mcp.context.failed")
        packet = failed_packet(["build_context_failed: see server logs"])
        token_count = 0
        trace_id = None
    else:
        metrics.increment(f"mcp.context.{packet['status']}")
    return {
        "context": _format_packet(settings.mcp_subject_id, packet),
        "facts": packet["facts"],
        "warnings": packet["warnings"],
        "status": packet["status"],
        "token_count": token_count,
        "trace_id": str(trace_id) if trace_id else None,
        "project_id": settings.mcp_project_id,
    }


@mcp.tool()
async def haki_capture(content: str, kind: str = "agent.observation") -> dict[str, Any]:
    """Memorise un fait durable du projet : decision technique, convention,
    erreur resolue. A appeler en FIN de tache. Ne jamais memoriser de
    secrets, tokens ou donnees ephemeres. La consolidation est synchrone en
    dev (HAKI_MCP_AUTOCONSOLIDATE), donc le fait est rappelable
    immediatement."""
    subject_id = settings.mcp_subject_id
    # Content-based idempotency key (no timestamp): an agent calling the
    # tool twice with the same memory does not create a duplicate event.
    digest = hashlib.sha256(f"{kind}\n{content}".encode("utf-8")).hexdigest()
    event = EventIn(
        org_id=settings.mcp_org_id,
        project_id=settings.mcp_project_id,
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
                project_id=settings.mcp_project_id,
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
        "project_id": settings.mcp_project_id,
    }


@mcp.tool()
async def haki_inspect(trace_id: str) -> dict[str, Any]:
    """Inspecte la trace d'un appel haki_context : quels faits ont ete
    inclus, exclus ou bloques, et pourquoi (reason_code). Preuve de
    provenance de la memoire servie."""
    async with async_session() as session:
        trace = await get_trace(
            session,
            trace_id=uuid.UUID(trace_id),
            project_id=settings.mcp_project_id,
            subject_id=settings.mcp_subject_id,
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
async def haki_forget(mode: str = "disable") -> dict[str, Any]:
    """Oublie la memoire du sujet configure pour ce serveur, dans ce projet.
    mode='disable' (reversible, les faits passent a disabled) ou 'delete'
    (effacement reel : faits, embeddings, evenements, traces). Retourne le
    recu d'effacement (forget_id) et les compteurs de ce qui a ete fait."""
    async with async_session() as session:
        receipt, counters = await ledger.forget(
            session,
            project_id=settings.mcp_project_id,
            subject_id=settings.mcp_subject_id,
            mode=mode,
        )
        await session.commit()
    return {
        "status": "ok",
        "mode": mode,
        "scope": receipt.scope,
        "forget_id": str(receipt.id),
        "project_id": settings.mcp_project_id,
        **counters,
    }


@mcp.tool()
async def haki_correct(
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
    que soit le chemin d'appel. Scope au projet configure pour ce serveur
    (HAKI_MCP_PROJECT_ID)."""
    async with async_session() as session:
        row, fact_status = await ledger.submit_feedback(
            session,
            project_id=settings.mcp_project_id,
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
        "project_id": settings.mcp_project_id,
    }
