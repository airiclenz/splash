"""The request fields every API reads alike, each read in one place."""

import hashlib

from . import protocol as wire
from .errors import APIError

# The priority a request names, by its lowercase wire name.
_PRIORITIES = {priority.name.lower(): priority for priority in wire.RequestPriority}


def generation(body):
    """Whether a generation request streams its answer, and the request flags
    its answer sets: RETURN_PROGRESS for return_progress, which only a stream
    takes. Every generation API reads these here first, stream before its
    other fields, so an invalid one is refused alike everywhere. The job
    adds the flags its generation options set (Frontend._generation_job)."""
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise APIError(400, "stream must be a boolean")
    progress = body.get("return_progress", False)
    if not isinstance(progress, bool) or (progress and not stream):
        raise APIError(
            400, "return_progress requires stream: true and must be a boolean"
        )
    flags = wire.RequestFlag.RETURN_PROGRESS if progress else wire.RequestFlag(0)
    return stream, flags


def priority(body):
    """The scheduling priority the request names; normal without one."""
    priority_name = body.get("priority", "normal")
    if not isinstance(priority_name, str) or priority_name not in _PRIORITIES:
        raise APIError(400, "priority must be foreground, normal, or background")
    return _PRIORITIES[priority_name]


def cache_salt(body):
    """The request's cache_salt as the engine keys its cache: 128 bits of
    its hash, zero without one. A request shares cached prompts only with
    requests of the same salt."""
    salt = body.get("cache_salt")
    if salt is None:
        return 0
    if not isinstance(salt, str) or not salt:
        raise APIError(400, "cache_salt must be a non-empty string")
    digest = hashlib.blake2b(
        salt.encode("utf-8", "surrogatepass"), digest_size=16, person=b"splash-salt"
    ).digest()
    return int.from_bytes(digest, "little")


def reasoning(body):
    """The reasoning effort a chat request sends, and whether it turns
    thinking on: its reasoning_effort, else the effort in OpenRouter's
    reasoning object ({"effort": ..., "enabled": ...}, which /v1/models'
    per-model reasoning invites and Responses also sends), where enabled
    false asks for "none". reasoning_effort wins over the object, whose
    other keys (Responses' summary, ...) are left alone. Returns
    (effort, enabled), None for each the request does not send; the effort
    is checked by request preparation alike from either field."""
    effort = body.get("reasoning_effort")
    sent = body.get("reasoning")
    if sent is None:
        return effort, None
    if not isinstance(sent, dict):
        raise APIError(400, "reasoning must be an object")
    enabled = sent.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise APIError(400, "reasoning enabled must be a boolean")
    object_effort = sent.get("effort")
    if enabled is not None and object_effort is not None:
        if (object_effort == "none") == enabled:
            raise APIError(400, "reasoning enabled contradicts its effort")
    if enabled is False:
        object_effort = "none"
    if effort is not None:
        return effort, None
    return object_effort, enabled
