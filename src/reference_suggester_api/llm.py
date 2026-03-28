from __future__ import annotations

import json
import re

from .models import QueryPlan


SYSTEM_PROMPT = """You are a scholarly retrieval assistant.
Convert one suspicious bibliography entry into compact JSON for academic search.
Do not invent citations.
Do not explain.
Output valid JSON only.

Required JSON keys:
topic: string
keywords: list of strings
synonyms: list of strings
query_list: list of 3 to 5 short scholarly search queries
likely_reference_type: one of ["journal_article","conference_paper","book","survey","arxiv","unknown"]
"""


class LocalLLMPlanner:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._tokenizer = None
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto',
            trust_remote_code=True,
        )
        self._pipe = pipeline(
            'text-generation',
            model=model,
            tokenizer=self._tokenizer,
            return_full_text=False,
        )

    def _fallback_plan(self, fallback_title: str) -> dict:
        fallback_title = fallback_title or 'unknown reference'
        return {
            'topic': fallback_title,
            'keywords': [w for w in re.findall(r'[A-Za-z]{4,}', fallback_title)[:5]],
            'synonyms': [],
            'query_list': [fallback_title],
            'likely_reference_type': 'unknown',
        }

    def _extract_json_lenient(self, text: str, fallback_title: str) -> dict:
        match = re.search(r'\{.*', text, flags=re.DOTALL)
        if not match:
            return self._fallback_plan(fallback_title)
        s = match.group(0).strip()
        open_braces = s.count('{')
        close_braces = s.count('}')
        if close_braces < open_braces:
            s += '}' * (open_braces - close_braces)
        try:
            obj = json.loads(s)
        except Exception:
            topic_m = re.search(r'"topic"\s*:\s*"([^"]*)"', s)
            keywords = re.findall(r'"keywords"\s*:\s*\[(.*?)\]', s, flags=re.DOTALL)
            synonyms = re.findall(r'"synonyms"\s*:\s*\[(.*?)\]', s, flags=re.DOTALL)
            queries = re.findall(r'"query_list"\s*:\s*\[(.*?)\]', s, flags=re.DOTALL)

            def parse_list(blob):
                return re.findall(r'"([^"]*)"', blob[0]) if blob else []

            obj = {
                'topic': topic_m.group(1) if topic_m else fallback_title,
                'keywords': parse_list(keywords),
                'synonyms': parse_list(synonyms),
                'query_list': parse_list(queries) or [fallback_title],
                'likely_reference_type': 'unknown',
            }
        for key in ['topic', 'keywords', 'synonyms', 'query_list', 'likely_reference_type']:
            if key not in obj:
                obj[key] = [] if key in {'keywords', 'synonyms', 'query_list'} else 'unknown'
        if not obj['query_list']:
            obj['query_list'] = [fallback_title]
        return obj

    def make_plan(self, original_source_reference, original_title, original_author=None, original_year=None) -> QueryPlan:
        self._load()
        user_prompt = f"""
Suspicious reference:
source_reference: {original_source_reference}
title: {original_title}
author: {original_author}
year: {original_year}

Return only JSON.
""".strip()
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = self._pipe(
            prompt,
            max_new_tokens=220,
            do_sample=False,
        )[0]['generated_text'].strip()
        data = self._extract_json_lenient(out, str(original_title or '').strip().rstrip(','))
        return QueryPlan(**data)
