from __future__ import annotations

import time
import requests


class OpenAlexRetriever:
    def __init__(self, timeout_seconds: int = 20):
        self.timeout_seconds = timeout_seconds
        self.base_url = 'https://api.openalex.org/works'

    def search(self, queries: list[str], per_query: int = 5) -> list[dict]:
        seen: dict[str, dict] = {}
        for q in queries[:5]:
            response = requests.get(
                self.base_url,
                params={'search': q, 'per_page': per_query},
                timeout=self.timeout_seconds,
            )
            if response.status_code != 200:
                continue
            for item in response.json().get('results', []):
                seen[item['id']] = item
            time.sleep(0.15)
        return list(seen.values())
