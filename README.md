# Reference Suggester API

Standalone FastAPI service for **reference repair and suggestion**. The service is designed to sit **after** the existing [Reference-Checker-API](https://github.com/AngshulMajumdar/Reference-Checker-API) and uses:

- the checker API for initial verification and artifact generation,
- a local instruct model for structured query generation,
- OpenAlex for candidate retrieval,
- sentence-transformer embeddings plus deterministic metadata scoring,
- conservative thresholding so weak matches are rejected instead of hallucinated.

The project intentionally keeps the original checker as a **separate service**. This repo calls it over HTTP.

---

## What the API does

Input:
- one uploaded `.pdf` or `.docx` paper

Pipeline:
1. call the upstream Reference Checker API,
2. fetch `changes.json` and related artifacts,
3. keep only the truly wrong references:
   - `unresolved`
   - `high_confidence_corrected`
4. generate compact search plans using a local LLM,
5. retrieve candidates from OpenAlex,
6. score candidates with hybrid semantic + metadata scoring,
7. apply strict filtering,
8. return three final outputs:
   - accepted replacements,
   - review queue,
   - no-valid-match list.

Output artifacts:
- `final_accept_replacements_dedup.csv`
- `final_review_queue_dedup.csv`
- `final_no_valid_found_dedup.csv`
- `final_apply_replacements.csv`
- `final_pipeline_summary.json`
- `llm_query_plans_final.json`
- `retrieval_scored_results.csv`
- `retrieval_strict_results.json`
- `reference_suggester_results.zip`

---

## Dependencies

### Python packages

Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The dependencies are:

- `pandas` — CSV processing and final report generation
- `requests` — HTTP calls to the checker API and OpenAlex
- `PyMuPDF` — PDF-side text support and future context extraction
- `python-docx` — ordinary DOCX parsing support
- `transformers`, `accelerate`, `sentencepiece`, `torch` — local LLM inference for query generation
- `sentence-transformers` — embedding-based semantic scoring
- `fastapi`, `uvicorn[standard]`, `python-multipart`, `pydantic` — API serving layer

### System dependency

For problematic `.docx` files, the pipeline converts DOCX to PDF through LibreOffice in headless mode.

Install separately:

```bash
sudo apt-get update
sudo apt-get install -y libreoffice
```

This is **not** a pip dependency and should not be added to `requirements.txt`.

---

## Directory structure

```text
reference_suggester_api_repo/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── run_api.py
├── start_api.sh
├── src/
│   └── reference_suggester_api/
│       ├── __init__.py
│       ├── app.py
│       ├── checker_client.py
│       ├── config.py
│       ├── doc_utils.py
│       ├── llm.py
│       ├── models.py
│       ├── pipeline.py
│       ├── retrieval.py
│       ├── scoring.py
│       └── utils.py
└── tests/
```

---

## Environment variables

The API works with sane defaults, but these variables can be overridden:

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
export REFSUGGEST_HOST=0.0.0.0
export REFSUGGEST_PORT=8010
export REFSUGGEST_STORAGE_DIR=./storage
export REFERENCE_CHECKER_BASE_URL=http://127.0.0.1:8000
export REFSUGGEST_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
export REFSUGGEST_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
export REFSUGGEST_CHECKER_TIMEOUT=900
export REFSUGGEST_OPENALEX_TIMEOUT=20
```

### Notes

- `REFERENCE_CHECKER_BASE_URL` must point to a running instance of the existing Reference Checker API.
- The default local LLM is `Qwen/Qwen2.5-3B-Instruct`.
- The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`.

---

## Running the API

First start the upstream checker API in another terminal or process.

Then start this API:

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
python run_api.py
```

or:

```bash
./start_api.sh
```

By default the service listens on:

```text
http://127.0.0.1:8010
```

---

## Endpoints

### Health

```http
GET /health
```

Returns current configuration summary.

### Suggest references

```http
POST /api/v1/suggest
```

Multipart form fields:
- `file` — required `.pdf` or `.docx`
- `checker_base_url` — optional override for the checker service

Response:
- job metadata
- checker job id
- artifact URLs
- summary counts

### Get job metadata

```http
GET /api/v1/jobs/{job_id}
```

### Download one artifact

```http
GET /api/v1/jobs/{job_id}/artifacts/{artifact_name}
```

### Download full bundle

```http
GET /api/v1/jobs/{job_id}/bundle
```

---

## Example cURL usage

### Submit one paper

```bash
curl -X POST "http://127.0.0.1:8010/api/v1/suggest" \
  -F "file=@/path/to/paper.pdf"
```

### Submit one DOCX and override checker URL

```bash
curl -X POST "http://127.0.0.1:8010/api/v1/suggest" \
  -F "file=@/path/to/paper.docx" \
  -F "checker_base_url=http://127.0.0.1:8000"
```

---

## Output philosophy

The service is intentionally conservative.

Final decisions are:

- `ACCEPT` — strong candidate, safe enough to auto-suggest
- `REVIEW` — plausible candidate, but should be checked by a human
- `NO_VALID` — do not force a replacement

This means the system may return fewer automatic replacements, but it avoids hallucinated papers.

---

## Practical limitations

- The service currently assumes one uploaded paper per API call.
- OpenAlex retrieval is used as the candidate source.
- The local LLM is used only for structured query generation, not for final acceptance.
- Very malformed references, missing titles, or extremely generic citations may still fall into `REVIEW` or `NO_VALID`.
- Difficult `.docx` files rely on LibreOffice conversion.

---

## Recommended production usage

Run the two services separately:

1. `Reference-Checker-API`
2. `Reference Suggester API`

That separation keeps the architecture clean:
- checker = verification engine
- suggester = retrieval + ranking engine

---

## Development notes

This repo intentionally mirrors the deployment feel of the checker repo:
- `run_api.py`
- `start_api.sh`
- `src/` layout
- artifact-based job storage

It is ready to be uploaded to GitHub and iterated further.


## Notes

- Heavy models are loaded lazily only when `/api/v1/suggest` is called. The `/health` endpoint should respond immediately after startup.
- Checker artifacts are downloaded into a guaranteed existing `checker_artifacts/` directory.
