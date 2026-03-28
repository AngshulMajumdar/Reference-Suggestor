from __future__ import annotations

from pathlib import Path
import requests


class CheckerClient:
    def __init__(self, base_url: str, timeout_seconds: int = 900):
        self.base_url = base_url.rstrip('/')
        self.timeout_seconds = timeout_seconds

    def verify(self, file_path: Path, *, include_semantic_scholar: bool = False, enable_hf_judge: bool = False) -> dict:
        with file_path.open('rb') as f:
            response = requests.post(
                f'{self.base_url}/api/v1/verify',
                files={'file': (file_path.name, f, 'application/octet-stream')},
                data={
                    'include_semantic_scholar': str(include_semantic_scholar).lower(),
                    'enable_hf_judge': str(enable_hf_judge).lower(),
                },
                timeout=self.timeout_seconds,
            )
        response.raise_for_status()
        return response.json()

    def fetch_artifact(self, artifact_path: str) -> bytes:
        response = requests.get(f'{self.base_url}{artifact_path}', timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.content
