"""Build a clean submission ZIP from working-tree code, optionally including the supplied corpus."""

import argparse
import hashlib
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.scan_secrets import _scan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-corpus", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("submission/census-insight-agent.zip"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = (
        subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .split("\0")
    )
    files: dict[str, Path] = {}
    for name in sorted(set(paths)):
        if not name:
            continue
        path = root / name
        if path.is_symlink() or not path.is_file():
            continue
        if name.startswith(("workspace/", "submission/", "data/processed/")):
            continue
        if path.name.startswith(".env") and path.name != ".env.example":
            raise ValueError("Refusing to package a local environment file")
        findings = _scan(name, path.read_bytes())
        if findings:
            raise ValueError(f"Submission secret scan failed: {findings}")
        files[name] = path
    if args.include_corpus:
        for folder, suffix in (("pdf", ".pdf"), ("markdown", ".md")):
            sources = sorted((root / "data/source" / folder).glob(f"*{suffix}"))
            if len(sources) != 3:
                raise ValueError(f"Expected three supplied {suffix} files")
            for path in sources:
                if path.is_symlink():
                    raise ValueError("Corpus symlinks are not packaged")
                files[path.relative_to(root).as_posix()] = path
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, f"census-insight-agent/{name}")
    checksum = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".zip.sha256").write_text(f"{checksum}  {args.output.name}\n")
    print(f"Created {args.output}: {len(files)} files, {args.output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
