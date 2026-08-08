from fastapi import APIRouter

from app.api.routes import (
    capture,
    cli_auth,
    conflicts,
    consolidate,
    context,
    feedback,
    forget,
    gateway,
    graph,
    health,
    inspection,
    keys,
    projects,
    stats,
    timeline,
)

api_router = APIRouter()
api_router.include_router(capture.router, prefix="/v1", tags=["capture"])
api_router.include_router(timeline.router, prefix="/v1", tags=["timeline"])
api_router.include_router(inspection.router, prefix="/v1", tags=["inspection"])
api_router.include_router(context.router, prefix="/v1", tags=["context"])
api_router.include_router(graph.router, prefix="/v1", tags=["graph"])
api_router.include_router(projects.router, prefix="/v1", tags=["projects"])
api_router.include_router(stats.router, prefix="/v1", tags=["stats"])
api_router.include_router(conflicts.router, prefix="/v1", tags=["conflicts"])
api_router.include_router(consolidate.router, prefix="/v1", tags=["consolidate"])
api_router.include_router(feedback.router, prefix="/v1", tags=["feedback"])
api_router.include_router(forget.router, prefix="/v1", tags=["forget"])
api_router.include_router(keys.router, prefix="/v1", tags=["keys"])
api_router.include_router(cli_auth.router, prefix="/v1", tags=["cli_auth"])
api_router.include_router(gateway.router, prefix="/gateway/v1", tags=["gateway"])
api_router.include_router(health.router, tags=["health"])
