#!/usr/bin/env bash
# Seed de démonstration pour la console (API réelle, FakeProvider).
# Crée : 1 fait actif, 1 fait remplacé (supersession), 1 conflit ouvert,
# des événements en timeline et une trace de contexte.
# NB : les JSON passent par stdin (--data-binary @-) — un `-d` en ligne de
# commande corrompt l'UTF-8 sous Git Bash/Windows.
set -euo pipefail

API="${HAKI_API_URL:-http://localhost:8100}"
KEY="${HAKI_KEY:?HAKI_KEY requis (hk_...)}"
PROJECT="prj_console"
SUBJECT="usr_demo"

consolidate() {
  curl -s -X POST "$API/v1/consolidate" -H "Authorization: Bearer $KEY" > /dev/null
  sleep 0.3
}

# 1 — préférence initiale : factures en français.
curl -s -X POST "$API/v1/capture" -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" --data-binary @- > /dev/null <<'EOF'
{"idempotency_key": "console-seed-1", "events": [{"org_id": "org_acme", "project_id": "prj_console", "subject_type": "user", "subject_id": "usr_demo", "kind": "conversation.message", "occurred_at": "2026-08-01T10:00:00Z", "payload": {"role": "user", "content": "Je préfère mes factures en français.", "mock_facts": [{"subject_id": "usr_demo", "predicate": "invoice_language", "value": {"language": "fr"}, "qualifiers": {}, "confidence": 0.9, "action": "create"}]}, "classification": ["customer-data"]}]}
EOF
consolidate

# 2 — changement d'avis : l'anglais remplace le français.
curl -s -X POST "$API/v1/capture" -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" --data-binary @- > /dev/null <<'EOF'
{"idempotency_key": "console-seed-2", "events": [{"org_id": "org_acme", "project_id": "prj_console", "subject_type": "user", "subject_id": "usr_demo", "kind": "conversation.message", "occurred_at": "2026-08-01T10:05:00Z", "payload": {"role": "user", "content": "Finalement, en anglais.", "mock_facts": [{"subject_id": "usr_demo", "predicate": "invoice_language", "value": {"language": "en"}, "qualifiers": {}, "confidence": 0.92, "action": "supersede", "supersedes_predicate": "invoice_language"}]}, "classification": ["customer-data"]}]}
EOF
consolidate

# 3 — canal de contact : email.
curl -s -X POST "$API/v1/capture" -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" --data-binary @- > /dev/null <<'EOF'
{"idempotency_key": "console-seed-3", "events": [{"org_id": "org_acme", "project_id": "prj_console", "subject_type": "user", "subject_id": "usr_demo", "kind": "conversation.message", "occurred_at": "2026-08-01T10:10:00Z", "payload": {"role": "user", "content": "Contactez-moi par email.", "mock_facts": [{"subject_id": "usr_demo", "predicate": "contact_channel", "value": {"channel": "email"}, "qualifiers": {}, "confidence": 0.85, "action": "create"}]}, "classification": ["customer-data"]}]}
EOF
consolidate

# 4 — contradiction : plutôt par chat (ouvre un conflit).
curl -s -X POST "$API/v1/capture" -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" --data-binary @- > /dev/null <<'EOF'
{"idempotency_key": "console-seed-4", "events": [{"org_id": "org_acme", "project_id": "prj_console", "subject_type": "user", "subject_id": "usr_demo", "kind": "conversation.message", "occurred_at": "2026-08-01T10:15:00Z", "payload": {"role": "user", "content": "Plutôt par chat en fait.", "mock_facts": [{"subject_id": "usr_demo", "predicate": "contact_channel", "value": {"channel": "chat"}, "qualifiers": {}, "confidence": 0.8, "action": "create"}]}, "classification": ["customer-data"]}]}
EOF
consolidate

# 5 — une trace de contexte (décisions included / blocked).
curl -s -X POST "$API/v1/context" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  --data-binary @- <<'EOF'
{"project_id": "prj_console", "subject_id": "usr_demo", "query": "dans quelle langue envoyer la facture ?", "purpose": "support"}
EOF
echo
echo "seed OK — $PROJECT / $SUBJECT"
