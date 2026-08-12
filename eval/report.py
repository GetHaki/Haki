"""Report writer: full JSON (audit) + readable Markdown (publication).

Every judged question is kept in the JSON with its verdict, tokens, latency
and cost — the report is auditable question by question. The Markdown cites
the exact frozen config and the dataset sha256.
"""

from __future__ import annotations

import json
from pathlib import Path


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _num(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def write_reports(
    results_dir: str | Path,
    dataset_name: str,
    run_id: str,
    config: dict,
    records: list[dict],
    summary: dict,
) -> tuple[Path, Path]:
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset_name}_{run_id}"
    json_path = results_path / f"{stem}.json"
    md_path = results_path / f"{stem}.md"

    json_path.write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "run_id": run_id,
                "config": config,
                "summary": summary,
                "questions": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    dataset_cfg = config.get("dataset", {})
    lines = [
        f"# Benchmark {dataset_name} — run {run_id}",
        "",
        "## Protocole (config figée)",
        "",
        f"- Dataset : `{dataset_cfg.get('file')}` (sha256 `{dataset_cfg.get('sha256', '?')[:16]}…`)",
        f"- Source : {dataset_cfg.get('url')}",
        f"- Subset : {config.get('selection', {}).get('subset', 'complet')} "
        f"(types : {', '.join(config.get('selection', {}).get('types') or ['tous'])})",
        f"- Modèle réponse : `{config.get('answer_model')}` — modèle juge : `{config.get('judge_model')}` "
        f"(temperature {config.get('temperature', 0)})",
        f"- Budget ContextPacket : {config.get('context_budget_tokens')} tokens — "
        f"baseline full-context tronquée à {config.get('baseline_max_context_tokens')} tokens",
        f"- Prompts : `{config.get('prompts', {}).get('answer')}` + `{config.get('prompts', {}).get('judge')}`",
        f"- Prix (USD / M tokens) : {json.dumps(config.get('prices_per_mtok', {}))}",
        "",
        "## Résultats",
        "",
        "| Système | Accuracy | Abstention (n) | Contradiction leakage (n) | Tokens contexte (moy.) | Latence p50 / p95 (ms) | Coût/requête (USD) | Coût ingestion (USD, n historiques) | Coût total (USD) |",
        "|---|---|---|---|---|---|---|---|---|"
        ,
    ]
    for system, stats in summary.items():
        if not stats.get("n"):
            continue
        latency = stats.get("latency_ms", {})
        lines.append(
            f"| {system} | {_pct(stats.get('accuracy'))} "
            f"| {_pct(stats.get('abstention', {}).get('accuracy'))} ({stats.get('abstention', {}).get('n', 0)}) "
            f"| {_pct(stats.get('contradiction_leakage', {}).get('rate'))} ({stats.get('contradiction_leakage', {}).get('n', 0)}) "
            f"| {_num(stats.get('context_tokens_mean'))} "
            f"| {_num(latency.get('p50'), 1)} / {_num(latency.get('p95'), 1)} "
            f"| {_num(stats.get('cost_per_query_usd'), 5)} "
            f"| {_num(stats.get('cost_ingest_usd'), 4)} ({stats.get('n_ingested', 0)}) "
            f"| {_num(stats.get('cost_usd'), 4)} |"
        )

    lines += ["", "### Accuracy par type de question", ""]
    types = sorted({t for stats in summary.values() for t in stats.get("per_type", {})})
    header = "| Type | " + " | ".join(summary.keys()) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(summary))
    for qtype in types:
        row = [qtype]
        for stats in summary.values():
            bucket = stats.get("per_type", {}).get(qtype)
            row.append(f"{_pct(bucket['accuracy'])} ({bucket['n']})" if bucket else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Détail par question", "", "| Question | Type | " + " | ".join(summary.keys()) + " |", "|---|---|" + "---|" * len(summary)]
    for record in records:
        row = [f"`{record['qid']}`", record["qtype"]]
        for system in summary.keys():
            result = record.get("systems", {}).get(system)
            row.append(result["label"] if result else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "---",
        "Rapport généré par le harnais `eval/` (protocole figé, baselines "
        "ré-exécutées dans le même protocole). Données complètes par question "
        f"dans `{json_path.name}`.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
