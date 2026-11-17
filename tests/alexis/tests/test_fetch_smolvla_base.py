"""P1.2: hash convention and cache filtering of scripts/train/fetch_smolvla_base.py."""

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fetch_smolvla_base", ROOT / "scripts" / "train" / "fetch_smolvla_base.py")
fetch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch)


def test_file_hash_matches_git_and_lfs(tmp_path):
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    assert fetch.file_hash(empty, lfs=False) == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"  # git's empty blob
    hello = tmp_path / "hello"
    hello.write_bytes(b"hello\n")
    assert fetch.file_hash(hello, lfs=False) == "ce013625030ba8dba906f756967f9e9ca394464a"  # `git hash-object`
    assert fetch.file_hash(hello, lfs=True) == hashlib.sha256(b"hello\n").hexdigest()


def test_cached_candidates_skip_migrated(tmp_path, monkeypatch):
    snaps = tmp_path / ".cache" / "huggingface" / "hub" / "models--lerobot--smolvla_base" / "snapshots"
    rev = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
    for d in (rev, rev + "_migrated"):
        (snaps / d).mkdir(parents=True)
        (snaps / d / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert fetch.cached_candidates("model.safetensors") == [snaps / rev / "model.safetensors"]
    assert fetch.cached_candidates("absent.json") == []
