from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Optional


class QueryPlan(BaseModel):
    topic: str
    keywords: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    query_list: list[str] = Field(default_factory=list)
    likely_reference_type: str = 'unknown'


class CandidateScore(BaseModel):
    openalex_id: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    type: Optional[str] = None
    source_name: Optional[str] = None
    semantic_score: float = 0.0
    title_overlap_original: float = 0.0
    title_overlap_topic: float = 0.0
    year_score: float = 0.0
    venue_type_score: float = 0.0
    author_hit: float = 0.0
    combined_metadata: float = 0.0
    total_score: float = 0.0


class SuggestionResult(BaseModel):
    file: str
    entry_key: str
    status_from_checker: str
    original_title: Optional[str] = None
    original_source_reference: Optional[str] = None
    llm_topic: Optional[str] = None
    decision_old: Optional[str] = None
    decision_strict: str
    suggested_title: Optional[str] = None
    suggested_year: Optional[int] = None
    suggested_doi: Optional[str] = None
    suggested_source_name: Optional[str] = None
    semantic_score: Optional[float] = None
    title_overlap_original: Optional[float] = None
    title_overlap_topic: Optional[float] = None
    year_score: Optional[float] = None
    author_hit: Optional[float] = None
    total_score: Optional[float] = None


class JobSummary(BaseModel):
    accepted_count: int
    review_count: int
    no_valid_count: int
    artifacts: dict[str, str]
    checker_job_id: Optional[str] = None
    converted_from_docx: bool = False


class StoredJobResponse(BaseModel):
    job_id: str
    filename: str
    created_at: str
    completed_at: str
    status: str
    checker_job_id: Optional[str] = None
    converted_from_docx: bool = False
    summary: dict[str, Any]
    artifacts: dict[str, str]
