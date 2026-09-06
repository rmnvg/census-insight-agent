"""Initialize only the shared queue directories for the unprivileged Linux executor."""

import os
from pathlib import Path


def main() -> None:
    root = Path("/execution")
    for path in [root, *(root / name for name in ("inbox", "processing", "results", "jobs"))]:
        path.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ValueError("Queue directories cannot be symlinks")
        path.chmod(0o770)
        os.chown(path, 10001, 10001)


if __name__ == "__main__":
    main()
