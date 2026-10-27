"""The build identity the server publishes, resolved once at startup.

A packaged release names its version in `release.json`; a source checkout
reports version `source` and the short commit of its tree, suffixed `-dirty`
when tracked files differ from that commit. The native build id comes from
the engine's own status. Resolution runs git at most twice and reads at most
one file; callers keep the result and never resolve per request.

Only the standard library is imported, so the packaged layout needs nothing
beyond this file. `dev/tools/build_identity.py` is unrelated: it computes the
native build id at build time.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# The program root: the checkout, or the unpacked release holding release.json.
ROOT = Path(__file__).resolve().parents[1]

# The version a source checkout reports in place of a release version.
SOURCE_VERSION = "source"

# The value of any identity field that could not be determined.
UNKNOWN = "unknown"

# The suffix a source commit carries when tracked files differ from it.
DIRTY_SUFFIX = "-dirty"

# The bound on each git call, so a wedged repository cannot stall startup.
GIT_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class BuildIdentity:
    """One build's `{version, commit, build_id}`; every field is non-empty."""

    version: str
    commit: str
    build_id: str

    @property
    def server_header(self):
        """The `Server:` header: `Splash/<version>` or `Splash/source-<commit>`."""
        if self.version == SOURCE_VERSION:
            return f"Splash/{SOURCE_VERSION}-{self.commit}"
        return f"Splash/{self.version}"

    def as_dict(self):
        """Return the JSON object `/health`, `/status` and `/version` carry."""
        return {
            "version": self.version,
            "commit": self.commit,
            "build_id": self.build_id,
        }


# The identity a server publishes before startup resolves one (tests use it).
UNRESOLVED = BuildIdentity(UNKNOWN, UNKNOWN, UNKNOWN)


def native_build_id(status):
    """Return the engine build id from a native status dict, or `unknown`.

    Any shape other than `{"identity": {"cache": {"build_id": <non-empty str>}}}`
    yields `unknown`, so a failed or stale status never breaks startup.
    """
    identity = status.get("identity") if isinstance(status, dict) else None
    cache = identity.get("cache") if isinstance(identity, dict) else None
    build_id = cache.get("build_id") if isinstance(cache, dict) else None
    return build_id if isinstance(build_id, str) and build_id else UNKNOWN


def resolve(build_id, root=ROOT, run=subprocess.run):
    """Resolve this program's build identity once.

    A `release.json` under `root` makes it a packaged release: its `version`
    is the version and the commit is `unknown` (a release carries no git
    tree). Otherwise the version is `source` and the commit comes from git
    via `run` (a `subprocess.run`-compatible callable). `build_id` is used as
    given, or `unknown` when empty.
    """
    build_id = build_id or UNKNOWN
    release = root / "release.json"
    if release.is_file():
        return BuildIdentity(_release_version(release), UNKNOWN, build_id)
    return BuildIdentity(SOURCE_VERSION, _source_commit(root, run), build_id)


def _release_version(release):
    """Return the `version` of a release.json, or `unknown` when unreadable."""
    try:
        version = json.loads(release.read_text())["version"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        _warn(f"release version unavailable: {error}")
        return UNKNOWN
    return str(version) or UNKNOWN


def _source_commit(root, run):
    """Return the short HEAD commit of `root`, `-dirty` when tracked files differ.

    Untracked files do not count as dirty, matching `git describe --dirty`.
    Any git failure yields `unknown` rather than a commit of uncertain state.
    """
    try:
        commit = _git(root, run, "rev-parse", "--short", "HEAD").strip()
        changes = _git(root, run, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError) as error:
        _warn(f"source commit unavailable: {error}")
        return UNKNOWN
    if not commit:
        return UNKNOWN
    return commit + DIRTY_SUFFIX if changes.strip() else commit


def _git(root, run, *arguments):
    """Run one bounded git command in `root` and return its stdout."""
    return run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=True,
        timeout=GIT_TIMEOUT_SECONDS,
    ).stdout


def _warn(message):
    """Report a resolution fallback on stderr; startup carries on with `unknown`."""
    print(f"Warning · build identity: {message}", file=sys.stderr, flush=True)
