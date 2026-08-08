from datetime import datetime

from pydantic import BaseModel


class ProjectOut(BaseModel):
    project_id: str
    active_facts: int
    subjects: int


class ProjectListResponse(BaseModel):
    projects: list[ProjectOut]


class SubjectOut(BaseModel):
    subject_id: str
    facts: int
    recalls: int
    last_seen: datetime


class SubjectListResponse(BaseModel):
    subjects: list[SubjectOut]
