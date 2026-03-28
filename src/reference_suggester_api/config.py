from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(slots=True)
class Settings:
    storage_dir: Path
    checker_base_url: str
    llm_model_name: str
    embedding_model_name: str
    checker_timeout_seconds: int
    openalex_timeout_seconds: int


def get_settings() -> Settings:
    return Settings(
        storage_dir=Path(os.environ.get('REFSUGGEST_STORAGE_DIR', './storage')).resolve(),
        checker_base_url=os.environ.get('REFERENCE_CHECKER_BASE_URL', 'http://127.0.0.1:8000').rstrip('/'),
        llm_model_name=os.environ.get('REFSUGGEST_LLM_MODEL', 'Qwen/Qwen2.5-3B-Instruct'),
        embedding_model_name=os.environ.get('REFSUGGEST_EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
        checker_timeout_seconds=int(os.environ.get('REFSUGGEST_CHECKER_TIMEOUT', '900')),
        openalex_timeout_seconds=int(os.environ.get('REFSUGGEST_OPENALEX_TIMEOUT', '20')),
    )
