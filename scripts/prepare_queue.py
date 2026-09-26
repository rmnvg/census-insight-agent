"""Initialize only the shared queue directories for the unprivileged Linux executor."""

import os
from pathlib import Path

EXECUTOR_UID = 10001


def main() -> None:
    root = Path("/execution")
    children = [root / name for name in ("inbox", "processing", "results", "jobs")]
    root.mkdir(parents=True, exist_ok=True)
    # This container keeps CHOWN and FOWNER but not DAC_OVERRIDE, so it can only enter
    # directories it owns. Take the queue root back before touching anything inside it (after an
    # earlier run it belongs to the executor), and hand it over again only at the very end.
    os.chown(root, 0, 0)
    root.chmod(0o700)
    for path in children:
        path.mkdir(exist_ok=True)
        if path.is_symlink():
            raise ValueError("Queue directories cannot be symlinks")
        path.chmod(0o770)
        os.chown(path, EXECUTOR_UID, EXECUTOR_UID)
    root.chmod(0o770)
    os.chown(root, EXECUTOR_UID, EXECUTOR_UID)


if __name__ == "__main__":
    main()
