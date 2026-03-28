from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .config import get_settings
from .doc_utils import save_upload
from .pipeline import SuggestionPipeline
from .utils import ensure_dir, utc_now, write_json

ALLOWED_SUFFIXES = {'.pdf', '.docx'}


def create_app() -> FastAPI:
    settings = get_settings()
    root = ensure_dir(settings.storage_dir)
    jobs_dir = ensure_dir(root / 'jobs')

    app = FastAPI(
        title='Reference Suggester API',
        version='1.0.0',
        description='Batch-safe reference suggestion API that uses an upstream checker, a local LLM, OpenAlex retrieval, and deterministic scoring.',
    )

    @app.get('/health')
    def health():
        return {
            'status': 'ok',
            'storage_dir': str(root),
            'checker_base_url': settings.checker_base_url,
            'llm_model_name': settings.llm_model_name,
            'embedding_model_name': settings.embedding_model_name,
        }

    @app.post('/api/v1/suggest')
    async def suggest_references(
        file: UploadFile = File(...),
        checker_base_url: str | None = Form(None),
    ):
        suffix = Path(file.filename or '').suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=400, detail=f'Unsupported file type: {suffix}')

        job_id = uuid.uuid4().hex
        job_dir = ensure_dir(jobs_dir / job_id)
        input_dir = ensure_dir(job_dir / 'input')
        output_dir = ensure_dir(job_dir / 'output')
        input_path = save_upload(file, input_dir / Path(file.filename).name)

        metadata = {
            'job_id': job_id,
            'filename': input_path.name,
            'created_at': utc_now(),
            'status': 'running',
        }
        write_json(job_dir / 'job.json', metadata)

        try:
            run_settings = get_settings()
            if checker_base_url:
                run_settings.checker_base_url = checker_base_url.rstrip('/')
            pipeline = SuggestionPipeline(run_settings)
            result = pipeline.run(input_path, output_dir)
            metadata.update({
                'completed_at': utc_now(),
                'status': 'completed',
                'checker_job_id': result['checker_job_id'],
                'converted_from_docx': result['converted_from_docx'],
                'summary': result['summary'],
                'artifacts': {
                    k: f'/api/v1/jobs/{job_id}/artifacts/{v}' if k != 'bundle_zip' else f'/api/v1/jobs/{job_id}/bundle'
                    for k, v in result['artifacts'].items()
                },
            })
            write_json(job_dir / 'job.json', metadata)
            return metadata
        except Exception as exc:
            metadata.update({
                'completed_at': utc_now(),
                'status': 'failed',
                'error': str(exc),
            })
            write_json(job_dir / 'job.json', metadata)
            raise HTTPException(status_code=500, detail=metadata)

    @app.get('/api/v1/jobs/{job_id}')
    def get_job(job_id: str):
        job_file = jobs_dir / job_id / 'job.json'
        if not job_file.exists():
            raise HTTPException(status_code=404, detail='Job not found')
        import json
        return json.loads(job_file.read_text(encoding='utf-8'))

    @app.get('/api/v1/jobs/{job_id}/artifacts/{artifact_name}')
    def get_artifact(job_id: str, artifact_name: str):
        artifact_path = jobs_dir / job_id / 'output' / artifact_name
        if not artifact_path.exists():
            raise HTTPException(status_code=404, detail='Artifact not found')
        return FileResponse(path=str(artifact_path), filename=artifact_path.name)

    @app.get('/api/v1/jobs/{job_id}/bundle')
    def get_bundle(job_id: str):
        bundle_path = jobs_dir / job_id / 'output' / 'reference_suggester_results.zip'
        if not bundle_path.exists():
            raise HTTPException(status_code=404, detail='Bundle not found')
        return FileResponse(path=str(bundle_path), filename=bundle_path.name)

    return app
