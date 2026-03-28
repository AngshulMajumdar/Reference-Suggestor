from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
import zipfile


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj) -> None:
    import json
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def zip_dir(src_dir: Path, dst_zip: Path) -> None:
    with zipfile.ZipFile(dst_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src_dir.rglob('*')):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(src_dir))


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
