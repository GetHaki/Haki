"""Cursor packaging (sprint 4): mcp.json snippet, install deeplink, Project Rule.

Pure generation, no I/O — the `haki mcp` CLI prints, tests assert.
"""

import base64
import json
import urllib.parse

DEFAULT_MCP_URL = "http://localhost:8100/mcp"


def mcp_config(url: str = DEFAULT_MCP_URL) -> dict:
    """The Cursor MCP server entry (streamable HTTP, no key in the file)."""
    return {"url": url}


def mcp_json_snippet(url: str = DEFAULT_MCP_URL) -> str:
    """Ready-to-paste .cursor/mcp.json content."""
    return json.dumps({"mcpServers": {"haki": mcp_config(url)}}, indent=2)


def deeplink(url: str = DEFAULT_MCP_URL, name: str = "haki") -> str:
    """`Add Haki to Cursor` one-click install link.

    cursor://anysphere.cursor-deeplink/mcp/install?name=haki&config=<cfg>
    where <cfg> is the base64 of the JSON server config, percent-encoded so
    the link survives copy/paste (base64 may contain +, / and =).
    """
    payload = json.dumps(mcp_config(url), separators=(",", ":")).encode("utf-8")
    config_b64 = base64.b64encode(payload).decode("ascii")
    return (
        "cursor://anysphere.cursor-deeplink/mcp/install"
        f"?name={urllib.parse.quote(name)}"
        f"&config={urllib.parse.quote(config_b64, safe='')}"
    )


def decode_deeplink_config(link: str) -> dict:
    """Inverse of deeplink(): extract and decode the config JSON (verification)."""
    params = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
    config_b64 = urllib.parse.unquote(params["config"][0])
    return json.loads(base64.b64decode(config_b64).decode("utf-8"))


def project_rule(url: str = DEFAULT_MCP_URL) -> str:
    """Content of .cursor/rules/haki.mdc (alwaysApply: project-wide usage
    conventions for the Haki tools)."""
    return f"""---
description: Memoire projet Haki — rappeler avant de planifier, capturer apres chaque tache
alwaysApply: true
---

# Memoire projet Haki

Ce projet est connecte a Haki (serveur MCP `{url}`), sa memoire
long-terme. Les outils `haki_*` donnent acces aux decisions, conventions,
preferences et erreurs deja resolues du projet.

## Avant de planifier ou de modifier du code

Appelle `haki_context` avec une requete decrivant la tache (ex. "quelles
conventions avant de modifier du code ?"). Injecte les faits retournes
(avec leurs dates et sources) dans ton raisonnement. Si un fait semble
obsolete ou faux, dis-le a l'utilisateur au lieu de le suivre aveuglement.

## En fin de tache

Appelle `haki_capture` pour memoriser ce qui a une valeur durable :

- une **decision technique** prise (et pourquoi) ;
- une **convention** du projet decouverte ou confirmee ;
- une **erreur resolue** (symptome + cause + correction) ;
- une **preference** explicite de l'utilisateur.

## Regles strictes

- Ne memorise que du **durable** : jamais de secrets, tokens, mots de
  passe, donnees personnelles ou contexte ephemere d'une session.
- Ne memorise pas le code lui-meme : Cursor garde son index de code ;
  Haki memorise les decisions et le contexte.
- Pour verifier POURQUOI un fait a ete servi, utilise `haki_inspect` avec
  le `trace_id` retourne par `haki_context`.
- Pour oublier une memoire a la demande de l'utilisateur, utilise
  `haki_forget` (`mode="disable"` par defaut, `mode="delete"` pour un
  effacement reel).
"""
