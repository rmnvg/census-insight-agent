#!/usr/bin/env python3
"""Report credential-like patterns by filename/category without exposing values."""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

PATTERNS: dict[str, re.Pattern[bytes]] = {
    "private_key": re.compile(rb"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    "google_api_key": re.compile(rb"AIza[0-9A-Za-z_-]{30,}"),
    "bearer_token": re.compile(rb"(?i)bearer\s+[A-Za-z0-9._~-]{20,}"),
    "service_account_secret": re.compile(
        rb'"(?:private_key|private_key_id)"\s*:\s*"(?!\s*")[^"\n]+"'
    ),
    "database_password": re.compile(
        rb"(?i)(?:database_url|db_password|database_password)\s*[=:]\s*[^\s\"']{8,}"
    ),
    "personal_absolute_path": re.compile(rb"/(?:Users|home)/[^/\s]+/"),
    "personal_email": re.compile(rb"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}

SELF = "scripts/scan_secrets.py"
SAFE_FILES = {".env.example"}


def _run(*args: str) -> bytes:
    return subprocess.run(args, check=True, stdout=subprocess.PIPE).stdout


def _scan(path: str, content: bytes) -> list[tuple[str, str]]:
    if path == SELF or path in SAFE_FILES:
        return []
    return [(path, category) for category, pattern in PATTERNS.items() if pattern.search(content)]


def scan_tracked() -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    in_repository = (
        bool(shutil.which("git"))
        and subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, check=False
        ).returncode
        == 0
    )
    if in_repository:
        paths = [
            item.decode("utf-8", "replace")
            for item in _run(
                "git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"
            ).split(b"\0")
            if item
        ]
    else:
        excluded = {
            ".git",
            ".venv",
            "data",
            "workspace",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "__pycache__",
            "submission",
        }
        paths = [
            str(path)
            for path in Path(".").rglob("*")
            if path.is_file()
            and not any(part in excluded for part in path.parts)
            and not path.name.endswith((".pyc", ".pyo"))
            and not (path.name == ".env" or path.name.startswith(".env."))
        ]
    for path in paths:
        try:
            content = Path(path).read_bytes()
        except OSError:
            continue
        findings.extend(_scan(path, content))
    return findings


def scan_history() -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    checked: set[bytes] = set()
    for line in _run("git", "rev-list", "--objects", "--all").splitlines():
        parts = line.split(b" ", 1)
        if len(parts) != 2 or parts[0] in checked:
            continue
        checked.add(parts[0])
        path = parts[1].decode("utf-8", "replace")
        kind = _run("git", "cat-file", "-t", parts[0].decode()).strip()
        if kind != b"blob":
            continue
        size = int(_run("git", "cat-file", "-s", parts[0].decode()))
        if size > 10_000_000:
            continue
        content = _run("git", "cat-file", "-p", parts[0].decode())
        findings.extend(_scan(path, content))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="also scan blobs reachable in Git")
    args = parser.parse_args()
    if args.history and not shutil.which("git"):
        print("FAIL history scan: Git is not installed in this environment.")
        return 1
    findings = scan_tracked()
    if args.history:
        findings.extend(scan_history())
    unique = sorted(set(findings))
    for path, category in unique:
        print(f"FAIL {category}: {path}")
    if unique:
        print(f"Secret scan failed: {len(unique)} filename/category finding(s).")
        return 1
    scope = "tracked files and Git history" if args.history else "tracked files"
    print(f"PASS secret scan: no credential-like values found in {scope}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
