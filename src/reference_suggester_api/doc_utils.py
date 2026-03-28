from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def convert_docx_to_pdf(docx_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        'soffice',
        '--headless',
        '--convert-to', 'pdf',
        '--outdir', str(out_dir),
        str(docx_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pdf_path = out_dir / f'{docx_path.stem}.pdf'
    if not pdf_path.exists():
        raise RuntimeError('LibreOffice conversion did not produce a PDF file.')
    return pdf_path


def save_upload(upload, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open('wb') as f:
        shutil.copyfileobj(upload.file, f)
    return dst
