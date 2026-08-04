"""Memory Policy Engine V1 (PRD — « Memory Policy Engine », US 38-42).

Deterministic, no LLM. Called BEFORE the action by capture/context/forget
and by the API key auth middleware. V1 rules, deliberately minimal (no
user-defined custom rules yet):

1. scope present — subject_id non vide sur chaque evenement capture
   (missing_scope, en coherence avec le Ledger) ;
2. key <-> project binding — une cle API ne voit que SON projet : tout
   project_id different dans le body ou la query est refuse en 403
   forbidden_scope, sans jamais reveler l'existence d'autres projets ;
3. purpose recommended on context — un appel context sans purpose est
   autorise mais journalise et signale (warning, pas erreur, en V1).

Every decision is logged as one structured JSON line (decision deny/warn
with its typed reason) so an incident review can replay what was refused
and why.
"""

import json
import logging
from collections.abc import Iterable

from app.errors import ApiError

logger = logging.getLogger("haki.policy")


def _log_decision(action: str, decision: str, reason: str, **extra: object) -> None:
    logger.info(
        "policy_decision %s",
        json.dumps(
            {"action": action, "decision": decision, "reason": reason, **extra},
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def check_project_scope(
    key_project_id: str,
    candidates: Iterable[str | None],
    *,
    action: str,
) -> None:
    """Rule 2 — a key only ever touches its own project.

    `candidates` are the project_ids found in the request (body and/or
    query). A mismatch is denied with 403 forbidden_scope and a generic
    message: no hint about the existence of other projects.
    """
    for candidate in candidates:
        if candidate and candidate != key_project_id:
            _log_decision(action, "deny", "forbidden_scope")
            raise ApiError(
                type="forbidden_scope",
                message="this API key is not authorized for the requested project",
                field="project_id",
                status_code=403,
            )


def check_capture_scope(events: list, *, action: str = "capture") -> None:
    """Rule 1 — subject scope present on every captured event."""
    for index, event in enumerate(events):
        if not event.subject_id:
            _log_decision(action, "deny", "missing_scope", field=f"events.{index}.subject_id")
            raise ApiError(
                type="missing_scope",
                message="subject_id is required on every event",
                field=f"events.{index}.subject_id",
            )


def context_purpose_warning(
    *, purpose: str | None, project_id: str, subject_id: str
) -> str | None:
    """Rule 3 — purpose recommended on context (warning only in V1)."""
    if purpose:
        return None
    _log_decision(
        "context", "warn", "missing_purpose", project_id=project_id, subject_id=subject_id
    )
    return "missing_purpose: 'purpose' is recommended on context calls (warning only in V1)"


def audit_forget(*, project_id: str, mode: str, scope: str) -> None:
    """Forget operations are always journaled (US 42 — audit)."""
    _log_decision("forget", "allow", "audited", project_id=project_id, mode=mode, scope=scope)
