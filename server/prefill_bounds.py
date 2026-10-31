"""The prefill bound a macOS display-stall abort taught a server, kept for
later starts on the same Mac model and macOS version.

The file holds one record, {"mac", "macos", "scope", "bound_ms"}. A missing
file is the normal case; a malformed or unreadable one warns and is ignored,
and one written on another Mac model or macOS version is ignored, since how
long macOS tolerates a GPU command depends on both. Nothing here fails a
start or a relaunch."""

import argparse
import contextlib
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

from . import serve_options
from .diagnostics import print_status

# Where the learned bound lives. recall and remember read it at call time,
# so tests patch it to a temporary directory.
PATH = Path.home() / "Library/Application Support/Splash/prefill-bound.json"

# sysctl answers at once; a wedged one must not hold the start.
MACHINE_QUERY_TIMEOUT_SECONDS = 5


def machine_key():
    """(Mac model, macOS version) of this Mac, None when either is unknown;
    a None key recalls and remembers nothing."""
    try:
        model = subprocess.run(
            ("sysctl", "-n", "hw.model"),
            capture_output=True,
            text=True,
            check=True,
            timeout=MACHINE_QUERY_TIMEOUT_SECONDS,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    macos = platform.mac_ver()[0]
    if not model or not macos:
        return None
    return model, macos


def recall(key):
    """(scope, bound_ms) learned on the key's Mac model and macOS version,
    None without a usable record."""
    if key is None:
        return None
    path = PATH
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as error:
        _warn_ignored(path, getattr(error, "strerror", None) or error)
        return None
    try:
        record = json.loads(text)
        if (record["mac"], record["macos"]) != tuple(key):
            return None
        if type(record["bound_ms"]) is not int:
            raise ValueError
        scope = serve_options.parse_prefill_bound_scope(record["scope"])
        bound_ms = serve_options.parse_prefill_bound(record["bound_ms"])
    except (ValueError, KeyError, TypeError, argparse.ArgumentTypeError):
        _warn_ignored(path, "malformed")
        return None
    return scope, bound_ms


def remember(key, scope, bound_ms):
    """Keep (scope, bound_ms) for later starts on the key's Mac model and
    macOS version, replacing the file whole; a failure warns once and keeps
    whatever was there."""
    path = PATH
    record = {"mac": key[0], "macos": key[1], "scope": scope, "bound_ms": bound_ms}
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".prefill-bound-", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as target:
                json.dump(record, target)
                target.flush()
                os.fsync(target.fileno())
            # A reader sees the old record or the new one, never a part.
            os.replace(temporary, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
    except OSError as error:
        print_status(
            f"Warning · cannot keep the learned prefill bound in {path}: "
            f"{error.strerror or error}",
            error=True,
        )


def _warn_ignored(path, reason):
    print_status(
        f"Warning · ignoring the learned prefill bound in {path}: {reason}",
        error=True,
    )
