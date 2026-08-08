from pydantic import BaseModel


class DailyCount(BaseModel):
    date: str  # YYYY-MM-DD
    count: int


class OverviewStatsResponse(BaseModel):
    active_facts: int
    events_this_week: list[DailyCount]
    recall_p50_ms: float | None
    recall_p99_ms: float | None
    hit_rate: float | None  # 0..1, None if there were zero recalls to measure
    context_tokens_served: int
    recall_count: int
