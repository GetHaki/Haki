"""injection_rate is the canonical name for what hit_rate has always
measured (share of context calls that injected at least one fact);
hit_rate stays as an alias for the console."""

from pydantic import BaseModel


class DailyCount(BaseModel):
    date: str  # YYYY-MM-DD
    count: int


class OverviewStatsResponse(BaseModel):
    active_facts: int
    events_this_week: list[DailyCount]
    recall_p50_ms: float | None
    recall_p99_ms: float | None
    hit_rate: float | None  # deprecated alias of injection_rate (console still reads it)
    context_tokens_served: int
    recall_count: int
    injection_rate: float | None  # share of recalls that served >= 1 fact, 0..1; None if no recalls


class HealthComponent(BaseModel):
    name: str  # injection | contradiction_integrity | conflict_hygiene | staleness
    value: float | None  # 0..1; None = not measurable yet (excluded from the score)
    weight: float  # nominal weight before renormalization over measured components


class HealthStatsResponse(BaseModel):
    window_days: int
    injection_rate: float | None
    fact_density: float | None
    write_rejection_rate: float | None
    rejection_breakdown: dict[str, int]
    contradiction_leakage: float | None
    staleness: float | None  # always None until the volatility horizon exists
    open_conflicts: int
    health_score: float | None  # 0..100; None when nothing is measurable
    components: list[HealthComponent]
    # Transparency counters: every ratio above is verifiable from these.
    traces_in_window: int
    packets_with_facts: int
    leaked_packets: int
    candidates_total: int
    active_facts: int
    events_total: int
