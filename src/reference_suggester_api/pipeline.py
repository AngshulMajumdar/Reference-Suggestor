from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
import pandas as pd

from .checker_client import CheckerClient
from .config import Settings
from .doc_utils import convert_docx_to_pdf
from .llm import LocalLLMPlanner
from .models import QueryPlan
from .retrieval import OpenAlexRetriever
from .scoring import HybridScorer
from .utils import write_json, zip_dir


class SuggestionPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.checker = CheckerClient(settings.checker_base_url, timeout_seconds=settings.checker_timeout_seconds)
        self.planner = LocalLLMPlanner(settings.llm_model_name)
        self.retriever = OpenAlexRetriever(settings.openalex_timeout_seconds)
        self.scorer = None

    def _call_checker_with_docx_fallback(self, input_path: Path, work_dir: Path) -> tuple[dict, Path, bool]:
        converted = False
        verify_path = input_path
        try:
            result = self.checker.verify(verify_path)
            return result, verify_path, converted
        except Exception:
            if input_path.suffix.lower() != '.docx':
                raise
        pdf_path = convert_docx_to_pdf(input_path, work_dir / 'converted')
        result = self.checker.verify(pdf_path)
        converted = True
        return result, pdf_path, converted

    def _fetch_checker_artifacts(self, checker_response: dict, out_dir: Path) -> dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts = checker_response.get('artifacts', {})
        fetched: dict[str, Path] = {}
        for key, artifact_path in artifacts.items():
            dst = out_dir / f'{key}{Path(artifact_path).suffix or ".bin"}'
            dst.write_bytes(self.checker.fetch_artifact(artifact_path))
            fetched[key] = dst
        return fetched

    @staticmethod
    def _extract_wrong_refs(changes: list[dict]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for item in changes:
            decision = item.get('decision', {}) or {}
            status = str(decision.get('status', '')).strip()
            if status == 'exact_verified':
                continue
            if status not in {'unresolved', 'high_confidence_corrected'}:
                continue
            orig = item.get('original', {}) or {}
            corr = item.get('corrected', {}) or {}
            cand = decision.get('selected_candidate', {}) or {}
            rows.append({
                'entry_key': item.get('entry_key'),
                'status': status,
                'confidence': decision.get('confidence'),
                'original_source_reference': orig.get('_source_reference'),
                'original_title': orig.get('title'),
                'original_author': orig.get('author'),
                'original_year': orig.get('year'),
                'candidate_title': cand.get('title'),
                'candidate_authors': '; '.join(cand.get('authors', []) if isinstance(cand.get('authors'), list) else []),
                'candidate_year': cand.get('year'),
                'candidate_venue': cand.get('venue'),
                'candidate_doi': cand.get('doi'),
                'selected_source': cand.get('source'),
                'n_changed_fields': sum(1 for k in set(orig.keys()).union(corr.keys()) if orig.get(k) != corr.get(k)),
            })
        return pd.DataFrame(rows)

    def _make_plan(self, row: pd.Series) -> dict[str, Any]:
        if row['status'] == 'high_confidence_corrected':
            title = str(row.get('candidate_title') or row.get('original_title') or '').strip()
            keywords = [w for w in __import__('re').findall(r'[A-Za-z]{4,}', title.lower())[:5]]
            plan = QueryPlan(
                topic=title,
                keywords=keywords,
                synonyms=[],
                query_list=[title],
                likely_reference_type='journal_article',
            )
        else:
            plan = self.planner.make_plan(
                original_source_reference=row.get('original_source_reference'),
                original_title=row.get('original_title'),
                original_author=row.get('original_author'),
                original_year=row.get('original_year'),
            )
        base = row.to_dict()
        base.update({
            'llm_topic': plan.topic,
            'llm_keywords': plan.keywords,
            'llm_synonyms': plan.synonyms,
            'llm_query_list': plan.query_list,
            'llm_likely_reference_type': plan.likely_reference_type,
        })
        return base

    def _retrieve_and_score(self, plan: dict[str, Any]) -> dict[str, Any]:
        if self.scorer is None:
            self.scorer = HybridScorer(self.settings.embedding_model_name)
        queries = list(dict.fromkeys([q for q in (plan.get('llm_query_list') or []) if str(q).strip()] + [str(plan.get('original_title') or '').strip()]))
        candidates = self.retriever.search(queries, per_query=5)
        scored: list[dict[str, Any]] = []
        for item in candidates:
            sem = self.scorer.semantic_score(plan, item)
            meta = self.scorer.metadata_score(plan, item)
            total = 0.60 * sem + 0.40 * meta['combined_metadata']
            scored_item = {
                'openalex_id': item.get('id'),
                'title': item.get('display_name'),
                'year': item.get('publication_year'),
                'doi': ((item.get('ids') or {}).get('doi') or item.get('doi')),
                'type': item.get('type'),
                'source_name': (((item.get('primary_location') or {}).get('source')) or {}).get('display_name'),
                'semantic_score': sem,
                **meta,
                'total_score': total,
            }
            scored.append(scored_item)
        scored = sorted(scored, key=lambda x: x['total_score'], reverse=True)
        top = scored[:5]
        filtered = []
        for c in top:
            if not self.scorer.is_valid_domain(c):
                continue
            if self.scorer.title_overlap(plan.get('original_title'), c.get('title')) < 0.3:
                continue
            filtered.append(c)
        if not filtered:
            decision_strict = 'NO_VALID'
            best = None
        else:
            best = sorted(filtered, key=lambda x: x['total_score'], reverse=True)[0]
            if best['total_score'] >= 0.80:
                decision_strict = 'ACCEPT'
            elif best['total_score'] >= 0.65:
                decision_strict = 'REVIEW'
            else:
                decision_strict = 'NO_VALID'
        return {
            **plan,
            'queries': queries,
            'decision_old': 'ACCEPT' if (top and top[0]['total_score'] >= 0.78) else ('REVIEW' if (top and top[0]['total_score'] >= 0.62) else 'NO_VALID_REPLACEMENT_FOUND'),
            'best': best,
            'top5': top,
            'decision_strict': decision_strict,
        }

    @staticmethod
    def _to_final_row(filename: str, item: dict[str, Any]) -> dict[str, Any]:
        best = item.get('best')
        row = {
            'file': filename,
            'entry_key': item.get('entry_key'),
            'status_from_checker': item.get('status'),
            'original_title': item.get('original_title'),
            'original_source_reference': item.get('original_source_reference'),
            'llm_topic': item.get('llm_topic'),
            'decision_old': item.get('decision_old'),
            'decision_strict': item.get('decision_strict'),
            'suggested_title': None if best is None else best.get('title'),
            'suggested_year': None if best is None else best.get('year'),
            'suggested_doi': None if best is None else best.get('doi'),
            'suggested_source_name': None if best is None else best.get('source_name'),
            'semantic_score': None if best is None else best.get('semantic_score'),
            'title_overlap_original': None if best is None else best.get('title_overlap_original'),
            'title_overlap_topic': None if best is None else best.get('title_overlap_topic'),
            'year_score': None if best is None else best.get('year_score'),
            'author_hit': None if best is None else best.get('author_hit'),
            'total_score': None if best is None else best.get('total_score'),
        }
        return row

    def run(self, input_path: Path, output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        checker_response, verify_path, converted_from_docx = self._call_checker_with_docx_fallback(input_path, output_dir)
        checker_artifacts = self._fetch_checker_artifacts(checker_response, output_dir / 'checker_artifacts')

        changes_path = checker_artifacts.get('changes_json')
        if changes_path is None:
            raise RuntimeError('Checker did not return changes_json artifact.')
        changes = json.loads(changes_path.read_text(encoding='utf-8'))

        wrong_df = self._extract_wrong_refs(changes)
        wrong_df.to_csv(output_dir / 'final_wrong_refs.csv', index=False)

        plans = [self._make_plan(row) for _, row in wrong_df.iterrows()]
        write_json(output_dir / 'llm_query_plans_final.json', plans)

        scored_results = [self._retrieve_and_score(plan) for plan in plans]
        write_json(output_dir / 'retrieval_strict_results.json', scored_results)

        scored_csv_rows = []
        final_rows = []
        for item in scored_results:
            best = item.get('best')
            scored_csv_rows.append({
                'entry_key': item.get('entry_key'),
                'original_title': item.get('original_title'),
                'decision': item.get('decision_old'),
                'decision_strict': item.get('decision_strict'),
                'best_title': None if best is None else best.get('title'),
                'best_year': None if best is None else best.get('year'),
                'best_doi': None if best is None else best.get('doi'),
                'best_score': None if best is None else best.get('total_score'),
            })
            final_rows.append(self._to_final_row(input_path.name, item))
        pd.DataFrame(scored_csv_rows).to_csv(output_dir / 'retrieval_scored_results.csv', index=False)

        final_df = pd.DataFrame(final_rows)
        accept_df = final_df[final_df['decision_strict'] == 'ACCEPT'].drop_duplicates(subset=['file', 'entry_key'])
        review_df = final_df[final_df['decision_strict'] == 'REVIEW'].drop_duplicates(subset=['file', 'entry_key'])
        novalid_df = final_df[final_df['decision_strict'] == 'NO_VALID'].drop_duplicates(subset=['file', 'entry_key'])

        accept_df.to_csv(output_dir / 'final_accept_replacements_dedup.csv', index=False)
        review_df.to_csv(output_dir / 'final_review_queue_dedup.csv', index=False)
        novalid_df.to_csv(output_dir / 'final_no_valid_found_dedup.csv', index=False)

        apply_df = accept_df[[
            'file', 'entry_key', 'original_source_reference', 'original_title', 'suggested_title',
            'suggested_year', 'suggested_doi', 'suggested_source_name', 'total_score'
        ]].copy()
        def _replacement_record(row):
            parts = []
            if pd.notna(row.get('suggested_title')):
                parts.append(str(row['suggested_title']))
            if pd.notna(row.get('suggested_year')):
                try:
                    parts.append(f"({int(float(row['suggested_year']))})")
                except Exception:
                    pass
            if pd.notna(row.get('suggested_source_name')):
                parts.append(str(row['suggested_source_name']))
            if pd.notna(row.get('suggested_doi')):
                parts.append(str(row['suggested_doi']))
            return ' | '.join(parts)
        apply_df['replacement_record'] = apply_df.apply(_replacement_record, axis=1)
        apply_df.to_csv(output_dir / 'final_apply_replacements.csv', index=False)

        summary = {
            'accepted_count': int(len(accept_df)),
            'review_count': int(len(review_df)),
            'no_valid_count': int(len(novalid_df)),
            'checker_job_id': checker_response.get('job_id'),
            'converted_from_docx': converted_from_docx,
        }
        write_json(output_dir / 'final_pipeline_summary.json', summary)

        bundle_zip = output_dir / 'reference_suggester_results.zip'
        zip_dir(output_dir, bundle_zip)

        return {
            'checker_job_id': checker_response.get('job_id'),
            'converted_from_docx': converted_from_docx,
            'summary': summary,
            'artifacts': {
                'final_accept_replacements': 'final_accept_replacements_dedup.csv',
                'final_review_queue': 'final_review_queue_dedup.csv',
                'final_no_valid_found': 'final_no_valid_found_dedup.csv',
                'final_apply_replacements': 'final_apply_replacements.csv',
                'final_pipeline_summary': 'final_pipeline_summary.json',
                'llm_query_plans': 'llm_query_plans_final.json',
                'retrieval_scored_results': 'retrieval_scored_results.csv',
                'retrieval_strict_results': 'retrieval_strict_results.json',
                'bundle_zip': 'reference_suggester_results.zip',
            },
        }
