import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path):
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def manifest(path, args, **extra):
    packages = {}
    for name in ('numpy', 'torch', 'transformers', 'datasets', 'peft', 'accelerate'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    inputs = {name: file_hash(value) for name, value in vars(args).items()
              if name in ('input', 'train_file') and value and Path(value).is_file()}
    write_json(path, {"created_at": datetime.now(timezone.utc).isoformat(),
                      "status": "new run of reconstructed implementation",
                      "arguments": vars(args), "input_sha256": inputs,
                      "packages": packages, **extra})
