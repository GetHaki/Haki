#!/usr/bin/env bash
# Diagnostic de l'environnement local Haki : Docker, conteneurs, Postgres,
# .env, migrations Alembic, API. Lecture seule et idempotent — ne modifie
# jamais rien, peut être relancé autant de fois que nécessaire.
#
# Usage : bash scripts/doctor.sh
#
# But : remplacer le diagnostic manuel (docker ps qui reste bloqué sans
# erreur claire, variables d'env oubliées, etc.) par un résultat lisible en
# une commande.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

API_URL="${HAKI_API_URL:-http://localhost:8100}"

if [ -t 1 ] && command -v tput >/dev/null 2>&1; then
  GREEN=$(tput setaf 2); RED=$(tput setaf 1); YELLOW=$(tput setaf 3)
  BOLD=$(tput bold); RESET=$(tput sgr0)
else
  GREEN=""; RED=""; YELLOW=""; BOLD=""; RESET=""
fi

FAILS=0

step() { printf '\n%s%s%s\n' "$BOLD" "$1" "$RESET"; }
ok()   { printf '%s✔%s %s\n' "$GREEN" "$RESET" "$1"; }
fail() { printf '%s✘%s %s\n' "$RED" "$RESET" "$1"; FAILS=$((FAILS + 1)); }
skip() { printf '%s⚠%s %s\n' "$YELLOW" "$RESET" "$1"; }
hint() { printf '   %s→%s %s\n' "$BOLD" "$RESET" "$1"; }

DOCKER_OK=0

# --- 1. Docker Desktop répond ----------------------------------------------
step "1. Docker Desktop"
if timeout 8 docker info >/dev/null 2>&1; then
  ok "Docker Desktop répond."
  DOCKER_OK=1
else
  fail "Docker Desktop ne répond pas (ou timeout de 8s dépassé)."
  hint "Ouvrez Docker Desktop et attendez l'icône 'Engine running', puis relancez ce script."
  hint "S'il reste bloqué : quittez-le complètement (clic droit sur l'icône -> Quit Docker Desktop), vérifiez l'espace disque disponible sur le disque système, puis relancez-le."
fi

# --- 2. Conteneurs Postgres et Redis ----------------------------------------
step "2. Conteneurs Postgres et Redis (docker compose ps)"
if [ "$DOCKER_OK" -eq 1 ]; then
  PS_JSON="$(docker compose ps --format json 2>/dev/null)"
  for SERVICE in postgres redis; do
    LINE="$(printf '%s\n' "$PS_JSON" | grep "\"Service\":\"$SERVICE\"")"
    if [ -z "$LINE" ]; then
      fail "$SERVICE : aucun conteneur trouvé pour ce projet."
      hint "Lancez : docker compose up -d"
    else
      STATE="$(printf '%s' "$LINE" | grep -o '"State":"[a-zA-Z]*"' | cut -d'"' -f4)"
      HEALTH="$(printf '%s' "$LINE" | grep -o '"Health":"[a-zA-Z]*"' | cut -d'"' -f4)"
      if [ "$STATE" = "running" ] && { [ -z "$HEALTH" ] || [ "$HEALTH" = "healthy" ]; }; then
        if [ -n "$HEALTH" ]; then
          ok "$SERVICE : running ($HEALTH)."
        else
          ok "$SERVICE : running."
        fi
      else
        fail "$SERVICE : état=${STATE:-inconnu} santé=${HEALTH:-inconnue}."
        hint "Consultez les logs : docker compose logs $SERVICE"
        hint "Ou (re)démarrez : docker compose up -d"
      fi
    fi
  done
else
  skip "Ignoré (Docker Desktop ne répond pas, voir étape 1)."
fi

# --- 3. Postgres accepte les connexions sur le port de docker-compose.yml --
step "3. Port Postgres (celui publié par docker-compose.yml, pas le 5432 par défaut)"
if [ "$DOCKER_OK" -eq 1 ]; then
  PG_PORT="$(docker compose port postgres 5432 2>/dev/null | cut -d: -f2 | tr -d '[:space:]')"
  if [ -z "$PG_PORT" ]; then
    fail "Impossible de déterminer le port publié pour le service postgres."
    hint "Vérifiez que le service tourne : docker compose up -d"
  else
    if timeout 3 bash -c "exec 3<>/dev/tcp/127.0.0.1/$PG_PORT" 2>/dev/null; then
      exec 3>&- 2>/dev/null
      ok "Postgres accepte les connexions TCP sur 127.0.0.1:$PG_PORT."
    else
      fail "Rien ne répond sur 127.0.0.1:$PG_PORT."
      hint "Vérifiez : docker compose ps   puis   docker compose logs postgres"
    fi
  fi
else
  skip "Ignoré (Docker Desktop ne répond pas, voir étape 1)."
fi

# --- 4. Fichier .env ----------------------------------------------------------
step "4. Fichier .env"
if [ -f "$REPO_ROOT/.env" ]; then
  ok ".env présent à la racine du dépôt."
else
  fail ".env absent."
  hint "Copiez le modèle : cp .env.example .env   (puis éditez les valeurs si besoin)"
fi

# --- 5. Migrations Alembic ----------------------------------------------------
step "5. Migrations Alembic (alembic current vs alembic heads)"
CURRENT_RAW="$(cd "$REPO_ROOT" && timeout 20 uv run alembic current 2>&1)"
CURRENT_STATUS=$?
HEADS_RAW="$(cd "$REPO_ROOT" && timeout 20 uv run alembic heads 2>&1)"
HEADS_STATUS=$?

if [ $CURRENT_STATUS -ne 0 ] || [ $HEADS_STATUS -ne 0 ]; then
  fail "Impossible d'interroger Alembic (DB injoignable, dépendances non installées, ou HAKI_MIGRATION_DATABASE_URL incorrecte)."
  hint "Vérifiez les dépendances : uv sync"
  hint "Vérifiez que Postgres tourne (étapes 1 à 3 ci-dessus) et que HAKI_MIGRATION_DATABASE_URL est correct."
else
  CURRENT_REV="$(printf '%s\n' "$CURRENT_RAW" | grep -v '^INFO' | grep -v '^$' | awk '{print $1}')"
  HEADS_REV="$(printf '%s\n' "$HEADS_RAW" | grep -v '^INFO' | grep -v '^$' | awk '{print $1}')"
  if [ -z "$CURRENT_REV" ]; then
    fail "Aucune migration appliquée sur cette base."
    hint "Lancez : uv run alembic upgrade head"
  elif [ "$CURRENT_REV" != "$HEADS_REV" ]; then
    fail "Base en retard : actuel=$CURRENT_REV, tête attendue=$HEADS_REV."
    hint "Lancez : uv run alembic upgrade head"
  else
    ok "Base à jour ($CURRENT_REV)."
  fi
fi

# --- 6. API /health -------------------------------------------------------------
step "6. API ($API_URL/health)"
HTTP_CODE="$(curl -s -o /dev/null -w '%{http_code}' -m 3 "$API_URL/health" 2>/dev/null)"
if [ "$HTTP_CODE" = "200" ]; then
  ok "L'API répond sur $API_URL/health."
else
  fail "L'API ne répond pas sur $API_URL/health (pas encore lancée, ou plantée)."
  hint "Lancez-la : uv run uvicorn app.main:app --port 8100"
fi

# --- Résumé --------------------------------------------------------------------
printf '\n%s%s%s\n' "$BOLD" "Résumé" "$RESET"
if [ "$FAILS" -eq 0 ]; then
  printf '%s✔ Tout est en ordre.%s\n' "$GREEN" "$RESET"
  exit 0
else
  printf '%s✘ %s vérification(s) en échec — voir le détail et les actions ci-dessus.%s\n' "$RED" "$FAILS" "$RESET"
  exit 1
fi
