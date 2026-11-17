r"""P1.2: put the official SmolVLA base weights in data/pretrained/smolvla_base, pinned and verified.

    .\.venv-learn\Scripts\python.exe scripts\train\fetch_smolvla_base.py

Source: the public Hugging Face repo lerobot/smolvla_base at a pinned revision (never a local
fine-tune). Each file is verified against the hub's hash for that revision (sha256 for LFS files,
git blob sha1 otherwise). Large files already in the local HF cache with the right hash are copied
instead of downloaded; the rest is downloaded. A SOURCE.json manifest records repo, revision and hashes.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO = "lerobot/smolvla_base"
REVISION = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"  # hub main on 2026-11-12
FILES = ["config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json",
         "policy_preprocessor_step_5_normalizer_processor.safetensors",
         "policy_postprocessor_step_0_unnormalizer_processor.safetensors"]


def file_hash(path: Path, lfs: bool) -> str:
    """sha256 of the content (LFS) or git blob sha1 (regular git file), as the hub reports them."""
    h = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def cached_candidates(name: str) -> list[Path]:
    """Same-named files in the official snapshots of the local HF cache (never '*_migrated' copies)."""
    snaps = Path.home() / ".cache" / "huggingface" / "hub" / "models--lerobot--smolvla_base" / "snapshots"
    if not snaps.is_dir():
        return []
    return [d / name for d in sorted(snaps.iterdir())
            if len(d.name) == 40 and all(c in "0123456789abcdef" for c in d.name) and (d / name).is_file()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/pretrained/smolvla_base")
    a = p.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"repo": REPO, "revision": REVISION, "files": {}}
    for name in FILES:
        s = remote[name]
        lfs = s.lfs is not None
        want = s.lfs.sha256 if lfs else s.blob_id
        dst = out / name
        how = "present"
        if not (dst.is_file() and file_hash(dst, lfs) == want):
            src = next((c for c in cached_candidates(name) if c.stat().st_size == s.size and file_hash(c, lfs) == want), None)
            if src:
                shutil.copyfile(src, dst)
                how = f"copied from cache {src.parent.name[:8]}"
            else:
                hf_hub_download(REPO, name, revision=REVISION, local_dir=out)
                how = "downloaded"
        got = file_hash(dst, lfs)
        if got != want:
            raise SystemExit(f"hash mismatch for {name}: {got} != {want}")
        manifest["files"][name] = {"size": s.size, "sha256" if lfs else "git_sha1": want, "source": how}
        print(f"[ok] {name:62s} {s.size:>11d}  {how}")
    (out / "SOURCE.json").write_text(json.dumps(manifest, indent=2))
    print(f"{out} verified against {REPO}@{REVISION[:8]}")


if __name__ == "__main__":
    main()
