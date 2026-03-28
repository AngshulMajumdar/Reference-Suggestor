from __future__ import annotations


class HybridScorer:
    def __init__(self, embedding_model_name: str):
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(embedding_model_name)

    @staticmethod
    def _safe_lower(x) -> str:
        return str(x).lower().strip() if x is not None else ''

    @classmethod
    def _norm_title(cls, x) -> str:
        return ' '.join(cls._safe_lower(x).replace(':', ' ').replace('-', ' ').replace(',', ' ').split())

    @classmethod
    def title_overlap(cls, a, b) -> float:
        sa = set(cls._norm_title(a).split())
        sb = set(cls._norm_title(b).split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / max(len(sa), 1)

    @staticmethod
    def year_score(y1, y2) -> float:
        try:
            y1 = int(float(y1))
            y2 = int(float(y2))
        except Exception:
            return 0.0
        d = abs(y1 - y2)
        if d == 0:
            return 1.0
        if d == 1:
            return 0.7
        if d == 2:
            return 0.4
        return 0.0

    @staticmethod
    def venue_hint_score(likely_type: str, item: dict) -> float:
        typ = str(item.get('type') or '').lower()
        src = (((item.get('primary_location') or {}).get('source')) or {})
        host_type = str(src.get('type') or '').lower()
        if likely_type == 'journal_article':
            return 1.0 if typ == 'article' else 0.3
        if likely_type == 'conference_paper':
            return 1.0 if typ in {'proceedings-article', 'article'} or host_type == 'conference' else 0.3
        if likely_type == 'book':
            return 1.0 if typ in {'book', 'book-chapter'} else 0.2
        if likely_type == 'arxiv':
            return 1.0 if 'arxiv' in str(src.get('display_name') or '').lower() else 0.2
        return 0.5

    def metadata_score(self, plan: dict, item: dict) -> dict:
        s1 = self.title_overlap(plan.get('original_title'), item.get('display_name'))
        s2 = self.title_overlap(plan.get('llm_topic'), item.get('display_name'))
        s3 = self.year_score(plan.get('original_year'), item.get('publication_year'))
        s4 = self.venue_hint_score(plan.get('llm_likely_reference_type', 'unknown'), item)
        cand_authors = [a.get('author', {}).get('display_name', '') for a in item.get('authorships', [])[:8]]
        orig_author = self._safe_lower(plan.get('original_author'))
        author_hit = 0.0
        if orig_author:
            author_tokens = [tok for tok in orig_author.replace('.', ' ').replace(',', ' ').split() if len(tok) >= 3]
            for a in cand_authors:
                al = self._safe_lower(a)
                if al and any(tok in al for tok in author_tokens):
                    author_hit = 1.0
                    break
        combined = 0.35 * s1 + 0.25 * s2 + 0.15 * s3 + 0.10 * s4 + 0.15 * author_hit
        return {
            'title_overlap_original': s1,
            'title_overlap_topic': s2,
            'year_score': s3,
            'venue_type_score': s4,
            'author_hit': author_hit,
            'combined_metadata': combined,
        }

    def semantic_score(self, plan: dict, item: dict) -> float:
        from sentence_transformers import util
        q = plan.get('llm_topic') or plan.get('original_title') or ''
        cand = item.get('display_name') or ''
        emb = self.embedding_model.encode([q, cand], convert_to_tensor=True, normalize_embeddings=True)
        return float(util.cos_sim(emb[0], emb[1]).item())

    @staticmethod
    def is_valid_domain(item: dict) -> bool:
        venue = str(item.get('source_name', '')).lower()
        title = str(item.get('title', '')).lower()
        good_keywords = [
            'signal', 'sparse', 'mimo', 'estimation', 'matrix',
            'compression', 'optimization', 'learning', 'information',
            'ieee', 'acm', 'springer', 'siam', 'quantum', 'comput', 'algorithm'
        ]
        return sum(kw in venue or kw in title for kw in good_keywords) >= 1
