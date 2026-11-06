"""Request-bound policy stack for dependency solving and background tasks.

This module provides a *per request* (or per WebSocket connection) stack of
policy frames. Each frame independently tunes four dimensions of the request
phase ("本段") it is pushed for:

* :class:`CacheSurface` -- how ``solve_dependencies`` interacts with the
  per-request ``dependency_cache`` (``Dependant.use_cache`` is still honored).
* :class:`OverrideSurface` -- whether ``dependency_overrides`` mappings are
  consulted, ignored, or frozen to a push-time snapshot.
* :class:`BackgroundAdmit` -- whether ``BackgroundTasks.add_task`` enqueues,
  rejects, or enqueues-but-suppresses execution after the response.
* :class:`GeneratorExitPolicy` -- whether generator dependencies are closed
  with the request/function ``AsyncExitStack``, held until the frame is
  popped, or closed immediately at frame pop.

The stack is stored in the ASGI ``scope`` of the current HTTP request /
WebSocket connection, so two concurrent ASGI requests can never share or
cross-contaminate each other's frames. There is deliberately no process
global stack. An empty stack (the default, with no frame ever pushed) is
behaviorally identical to plain FastAPI.

This module is intentionally decoupled from response serialization
(``response_model_*`` / ``exclude_none`` / ``by_alias`` / validation mode),
from OpenAPI generation and from the ``TestClient`` transport.
"""

from __future__ import annotations

import itertools
import sys
from collections.abc import AsyncIterator, Mapping, MutableMapping
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fastapi.types import DependencyCacheKey
from starlette.requests import HTTPConnection

__all__ = [
    "BackgroundAdmit",
    "BackgroundTaskRejectedError",
    "BackgroundTasksView",
    "CacheSurface",
    "EffectivePolicyView",
    "GeneratorExitPolicy",
    "OverrideSurface",
    "PolicyFrame",
    "RequestPolicy",
    "RequestPolicyStackError",
    "get_background_tasks_view",
    "get_frozen_cache_keys",
    "get_policy_view",
    "pop_request_policy",
    "push_request_policy",
    "request_policy",
]


# ---------------------------------------------------------------------------
# Public enums -- the four dimensions. They are intentionally distinct enums
# and must never be collapsed into a single boolean flag.
# ---------------------------------------------------------------------------


class CacheSurface(str, Enum):
    """How a frame interacts with the per-request ``dependency_cache``."""

    honor_cache = "honor_cache"
    #: Solves in this frame must not hit cache entries that existed before the
    #: frame; dependencies are observably re-invoked. Values solved inside the
    #: frame remain readable by later sub-dependencies in the same frame.
    bypass_cache = "bypass_cache"
    #: Keys written to the cache while this frame is active are frozen: other
    #: request phases (including concurrent requests sharing a cache object)
    #: may not overwrite or clear them. Later sub-dependencies in the same
    #: request still read them per normal cache semantics.
    freeze_solved = "freeze_solved"


class OverrideSurface(str, Enum):
    """How a frame consults ``dependency_overrides`` mappings."""

    honor_overrides = "honor_overrides"
    #: Solves in this frame ignore overrides and always call the original
    #: dependency callable.
    ignore_overrides = "ignore_overrides"
    #: Solves in this frame use a snapshot of the overrides mapping taken when
    #: the frame was pushed; later add/remove/change on the mapping does not
    #: affect the frame.
    freeze_map = "freeze_map"


class BackgroundAdmit(str, Enum):
    """How a frame treats ``BackgroundTasks.add_task`` and post-response run."""

    honor_queue = "honor_queue"
    #: ``add_task`` in this frame must measurably fail (it raises) and must not
    #: enqueue the task. Tasks already queued before the frame are preserved.
    reject_add = "reject_add"
    #: Tasks may be enqueued in this frame, but they are skipped when the
    #: background queue runs after the response and remain observable in the
    #: queue. The queue is never cleared to fake this.
    suppress_run = "suppress_run"


class GeneratorExitPolicy(str, Enum):
    """When generator/async-generator dependencies registered in a frame exit."""

    honor_astack = "honor_astack"
    #: Generators entered in this frame are not closed when the response ends;
    #: they are closed when the frame is popped (also on exception unwind).
    hold_until_pop = "hold_until_pop"
    #: Generators entered in this frame are closed immediately when the frame
    #: is popped, even if the response has not finished yet.
    force_close_now = "force_close_now"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RequestPolicyStackError(RuntimeError):
    """Misuse of the request policy stack (e.g. out-of-order pop)."""


class BackgroundTaskRejectedError(RuntimeError):
    """Raised by ``BackgroundTasks.add_task`` when the active frame rejects it."""


# ---------------------------------------------------------------------------
# Policy + frame + stack
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RequestPolicy:
    """A full set of the four dimensions for one frame."""

    cache_surface: CacheSurface = CacheSurface.honor_cache
    override_surface: OverrideSurface = OverrideSurface.honor_overrides
    bg_admit: BackgroundAdmit = BackgroundAdmit.honor_queue
    gen_exit: GeneratorExitPolicy = GeneratorExitPolicy.honor_astack


@dataclass(slots=True)
class PolicyFrame:
    """One pushed frame. Frames are popped LIFO by :func:`pop_request_policy`."""

    policy: RequestPolicy
    stack: _RequestPolicyStack
    depth: int
    #: provider object id -> immutable snapshot mapping, for ``freeze_map``.
    override_snapshots: dict[int, Mapping[Any, Any]] = field(default_factory=dict)
    #: Cache keys solved while this (bypass) frame is active; readable within
    #: the frame even though pre-frame cache entries are ignored.
    solved_keys: set[DependencyCacheKey] = field(default_factory=set)
    #: Private exit stack holding generator dependencies captured by this
    #: frame. It is intentionally *not* attached to the request/function
    #: stacks, so response-end teardown does not close them.
    astack: AsyncExitStack | None = None
    held_generator_count: int = 0

    @property
    def effective_cache_surface(self) -> CacheSurface:
        return self.policy.cache_surface

    @property
    def effective_override_surface(self) -> OverrideSurface:
        return self.policy.override_surface

    @property
    def effective_bg_admit(self) -> BackgroundAdmit:
        return self.policy.bg_admit

    @property
    def effective_gen_exit(self) -> GeneratorExitPolicy:
        return self.policy.gen_exit


_SCOPE_KEY = "fastapi_request_policy_stack"
_FROZEN_ATTR = "_fastapi_frozen"

# Per-connection contextvar mirror. It points at the scope-bound stack so
# nested calls that do not have the Request/WebSocket at hand can push frames.
# It is never a process-wide stack: each ASGI request resets it for its own
# scope-bound object.
_current_stack_var: ContextVar[_RequestPolicyStack | None] = ContextVar(
    "fastapi_request_policy_stack", default=None
)

_stack_tokens = itertools.count(1)

# Fallback freeze registry for dependency cache objects passed in by callers
# that are plain dicts (not our _PolicyCache). Framework-created per-request
# caches carry their own metadata and never touch this.
_foreign_frozen: dict[int, dict[DependencyCacheKey, int]] = {}


class _PolicyCache(dict[DependencyCacheKey, Any]):
    """Per-request dependency cache carrying freeze metadata.

    Behaviorally a plain ``dict`` when no frame freezes anything.
    """

    __slots__ = ("_fastapi_frozen",)

    _fastapi_frozen: dict[DependencyCacheKey, int]

    def __init__(self) -> None:
        super().__init__()
        self._fastapi_frozen = {}


def _frozen_map(cache: Any, *, create: bool = False) -> dict[Any, int] | None:
    own = getattr(cache, _FROZEN_ATTR, None)
    if isinstance(own, dict):
        return own
    if create:
        return _foreign_frozen.setdefault(id(cache), {})
    return None


def _frozen_owner(cache: Any, key: Any) -> int | None:
    frozen = _frozen_map(cache)
    if frozen is None:
        return None
    return frozen.get(key)


def _mark_frozen(cache: Any, key: Any, owner: int) -> None:
    frozen = getattr(cache, _FROZEN_ATTR, None)
    if isinstance(frozen, dict):
        frozen[key] = owner
        return
    _foreign_frozen.setdefault(id(cache), {})[key] = owner


def get_frozen_cache_keys(dependency_cache: Mapping[Any, Any]) -> frozenset[Any]:
    """Return the cache keys currently frozen on a dependency cache."""
    frozen = _frozen_map(dependency_cache)
    if frozen is None:
        return frozenset()
    return frozenset(frozen)


class _RequestPolicyStack:
    """The per-scope stack. One instance per HTTP request / WS connection."""

    __slots__ = ("token", "frames", "app")

    def __init__(self, app: Any = None) -> None:
        self.token = next(_stack_tokens)
        self.frames: list[PolicyFrame] = []
        # The ASGI app (overrides provider) captured when the stack is bound,
        # so freeze_map pushes without a connection handle can still snapshot
        # the mapping at push time.
        self.app = app

    def push(self, policy: RequestPolicy, scope: Mapping[str, Any]) -> PolicyFrame:
        frame = PolicyFrame(policy=policy, stack=self, depth=len(self.frames) + 1)
        # Eagerly snapshot the application-level overrides mapping so that
        # mutations between push and the first solve cannot leak into the
        # freeze_map frame.
        if policy.override_surface is OverrideSurface.freeze_map:
            app = scope.get("app") if scope else None
            if app is None:
                app = self.app
            provider_overrides = getattr(app, "dependency_overrides", None)
            if app is not None and provider_overrides is not None:
                frame.override_snapshots[id(app)] = dict(provider_overrides)
        self.frames.append(frame)
        return frame

    def pop(self, frame: PolicyFrame) -> PolicyFrame:
        if not self.frames or self.frames[-1] is not frame:
            raise RequestPolicyStackError(
                "Request policy frames must be popped in LIFO order"
            )
        self.frames.pop()
        return frame

    def top(self) -> PolicyFrame | None:
        return self.frames[-1] if self.frames else None


# ---------------------------------------------------------------------------
# Read-only effective view
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EffectivePolicyView:
    """Read-only effective policy for the current solving point.

    Dimensions fall back to their ``honor_*`` default when the stack is
    empty; an inner frame takes precedence over outer frames and popping it
    restores the outer values.
    """

    cache_surface: CacheSurface
    override_surface: OverrideSurface
    bg_admit: BackgroundAdmit
    gen_exit: GeneratorExitPolicy
    depth: int
    held_generator_count: int
    stack_token: int | None
    _frame: PolicyFrame | None
    _stack: _RequestPolicyStack | None

    def override_mapping_for(self, provider: Any) -> Mapping[Any, Any]:
        """Effective ``dependency_overrides`` mapping for a provider."""
        if provider is None:
            return {}
        if self.override_surface is OverrideSurface.ignore_overrides:
            return {}
        live = getattr(provider, "dependency_overrides", None)
        if not isinstance(live, Mapping) or not live:
            return {}
        if (
            self.override_surface is OverrideSurface.freeze_map
            and self._frame is not None
        ):
            snapshots = self._frame.override_snapshots
            snapshot = snapshots.get(id(provider))
            if snapshot is None:
                # Lazily freeze a provider first seen while solving (e.g. a
                # router-level overrides provider different from the app).
                snapshot = dict(live)
                snapshots[id(provider)] = snapshot
            return snapshot
        return live

    def cache_read_allowed(
        self, key: DependencyCacheKey, *, dependant_use_cache: bool = True
    ) -> bool:
        """Whether an existing cache entry may satisfy a solve."""
        if not dependant_use_cache:
            return False
        if self.cache_surface is CacheSurface.bypass_cache and self._frame is not None:
            return key in self._frame.solved_keys
        return True

    def captures_generators(self) -> bool:
        """Whether generators solved at this point go on the frame's stack."""
        return (
            self.gen_exit
            in (
                GeneratorExitPolicy.hold_until_pop,
                GeneratorExitPolicy.force_close_now,
            )
            and self._frame is not None
        )

    def store_solved(
        self,
        dependency_cache: dict[DependencyCacheKey, Any],
        key: DependencyCacheKey,
        solved: Any,
    ) -> None:
        """Write a freshly solved value honoring bypass/freeze semantics."""
        frame = self._frame
        stack = self._stack
        if key in dependency_cache:
            owner = _frozen_owner(dependency_cache, key)
            # A frozen entry can only be overwritten within the owning
            # request/connection; concurrent requests sharing the cache must
            # not overwrite or clear it.
            if (
                frame is not None
                and stack is not None
                and self.cache_surface is CacheSurface.bypass_cache
                and (owner is None or owner == stack.token)
            ):
                dependency_cache[key] = solved
        else:
            dependency_cache[key] = solved
            if (
                frame is not None
                and stack is not None
                and self.cache_surface is CacheSurface.freeze_solved
            ):
                _mark_frozen(dependency_cache, key, stack.token)
        if frame is not None and self.cache_surface is CacheSurface.bypass_cache:
            frame.solved_keys.add(key)


_DEFAULT_VIEW = EffectivePolicyView(
    cache_surface=CacheSurface.honor_cache,
    override_surface=OverrideSurface.honor_overrides,
    bg_admit=BackgroundAdmit.honor_queue,
    gen_exit=GeneratorExitPolicy.honor_astack,
    depth=0,
    held_generator_count=0,
    stack_token=None,
    _frame=None,
    _stack=None,
)


def _view_for_stack(stack: _RequestPolicyStack | None) -> EffectivePolicyView:
    if stack is None:
        return _DEFAULT_VIEW
    frame = stack.top()
    if frame is None:
        return _DEFAULT_VIEW
    held = sum(f.held_generator_count for f in stack.frames if f.astack is not None)
    return EffectivePolicyView(
        cache_surface=frame.effective_cache_surface,
        override_surface=frame.effective_override_surface,
        bg_admit=frame.effective_bg_admit,
        gen_exit=frame.effective_gen_exit,
        depth=len(stack.frames),
        held_generator_count=held,
        stack_token=stack.token,
        _frame=frame,
        _stack=stack,
    )


# ---------------------------------------------------------------------------
# Scope binding
# ---------------------------------------------------------------------------


def _stack_for_scope(
    scope: Mapping[str, Any], *, create: bool
) -> _RequestPolicyStack | None:
    stack = scope.get(_SCOPE_KEY)
    if isinstance(stack, _RequestPolicyStack):
        return stack
    if create:
        stack = _RequestPolicyStack(app=scope.get("app"))
        # Scope is a regular mutable ASGI dict in real requests.
        scope[_SCOPE_KEY] = stack  # type: ignore[index]
        return stack
    return None


def ensure_request_policy_stack(
    scope: MutableMapping[str, Any],
) -> _RequestPolicyStack:
    """Get or create the per-scope stack. Called by the routing wrappers."""
    stack = _stack_for_scope(scope, create=False)
    if stack is None:
        stack = _RequestPolicyStack(app=scope.get("app"))
        scope[_SCOPE_KEY] = stack
    return stack


def _resolve_stack(
    connection: HTTPConnection | None, *, create: bool
) -> _RequestPolicyStack | None:
    if connection is not None:
        return _stack_for_scope(connection.scope, create=create)
    return _current_stack_var.get()


# ---------------------------------------------------------------------------
# Public push/pop API
# ---------------------------------------------------------------------------


def push_request_policy(
    connection: HTTPConnection | None = None,
    policy: RequestPolicy | None = None,
    *,
    cache_surface: CacheSurface | None = None,
    override_surface: OverrideSurface | None = None,
    bg_admit: BackgroundAdmit | None = None,
    gen_exit: GeneratorExitPolicy | None = None,
) -> PolicyFrame:
    """Push a policy frame on the current request/connection stack.

    ``connection`` is the ``Request``/``WebSocket``; when omitted, the stack
    bound to the current request via context propagation is used. Raises
    :class:`RequestPolicyStackError` when no request-bound stack exists (there
    is intentionally no process global fallback).

    Dimensions left as ``None`` inherit the currently effective value, so a
    frame may change a single dimension (e.g. only ``bg_admit``) without
    resetting the others. Pass an explicit :class:`RequestPolicy` to specify
    all four dimensions at once.
    """
    stack = _resolve_stack(connection, create=True if connection else False)
    if stack is None:
        raise RequestPolicyStackError(
            "No request-bound policy stack. Push from middleware, a "
            "dependency or an endpoint with the Request/WebSocket available."
        )
    if policy is None:
        current = _view_for_stack(stack)
        policy = RequestPolicy(
            cache_surface=cache_surface or current.cache_surface,
            override_surface=override_surface or current.override_surface,
            bg_admit=bg_admit or current.bg_admit,
            gen_exit=gen_exit or current.gen_exit,
        )
    return stack.push(policy, connection.scope if connection is not None else {})


async def pop_request_policy(
    frame: PolicyFrame, connection: HTTPConnection | None = None
) -> None:
    """Pop a frame (LIFO) and close generators it captured.

    For ``hold_until_pop`` this is the moment the held generators exit; for
    ``force_close_now`` the generators observably close here even while the
    response is still in flight.
    """
    stack = frame.stack
    stack.pop(frame)
    if frame.astack is not None:
        astack = frame.astack
        frame.astack = None
        exc_info = sys.exc_info()
        try:
            if exc_info[0] is not None:
                await astack.__aexit__(*exc_info)
            else:
                await astack.__aexit__(None, None, None)
        finally:
            frame.held_generator_count = 0


@asynccontextmanager
async def request_policy(
    connection: HTTPConnection | None = None,
    policy: RequestPolicy | None = None,
    *,
    cache_surface: CacheSurface | None = None,
    override_surface: OverrideSurface | None = None,
    bg_admit: BackgroundAdmit | None = None,
    gen_exit: GeneratorExitPolicy | None = None,
) -> AsyncIterator[PolicyFrame]:
    """Async context manager: push on enter, pop (even on exception) on exit."""
    frame = push_request_policy(
        connection,
        policy,
        cache_surface=cache_surface,
        override_surface=override_surface,
        bg_admit=bg_admit,
        gen_exit=gen_exit,
    )
    try:
        yield frame
    finally:
        await pop_request_policy(frame, connection)


def get_policy_view(
    connection: HTTPConnection | None = None,
) -> EffectivePolicyView:
    """Return the read-only effective view at the current point.

    Empty stack -> the default view, which is exactly the unmodified
    dependency/background behavior.
    """
    stack = _resolve_stack(connection, create=False)
    return _view_for_stack(stack)


async def _ensure_frame_astack(frame: PolicyFrame) -> AsyncExitStack:
    if frame.astack is None:
        frame.astack = AsyncExitStack()
        await frame.astack.__aenter__()
    frame.held_generator_count += 1
    return frame.astack


# ---------------------------------------------------------------------------
# Background tasks policy state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _BackgroundPolicyState:
    """Attached to a FastAPI ``BackgroundTasks`` instance created by the solver."""

    stack: _RequestPolicyStack
    _suppressed: set[int] = field(default_factory=set)
    executed: int = 0
    skipped: int = 0

    def current_admit(self) -> BackgroundAdmit:
        return _view_for_stack(self.stack).bg_admit

    def mark_suppressed(self, task: Any) -> None:
        self._suppressed.add(id(task))

    def is_suppressed(self, task: Any) -> bool:
        return id(task) in self._suppressed

    def record_executed(self) -> None:
        self.executed += 1

    def record_skipped(self) -> None:
        self.skipped += 1

    @property
    def suppressed_count(self) -> int:
        return len(self._suppressed)


def bind_background_tasks(tasks: Any, scope: Mapping[str, Any]) -> None:
    """Bind a solver-created BackgroundTasks instance to its request stack."""
    if getattr(tasks, "_fastapi_policy_state", None) is not None:
        return
    stack = _stack_for_scope(scope, create=False)
    if stack is not None:
        tasks._fastapi_policy_state = _BackgroundPolicyState(stack=stack)


@dataclass(frozen=True, slots=True)
class BackgroundTasksView:
    """Read-only view of a BackgroundTasks queue merged with policy state."""

    queued: int
    suppressed: int
    executed: int
    skipped: int


def get_background_tasks_view(tasks: Any) -> BackgroundTasksView | None:
    state = getattr(tasks, "_fastapi_policy_state", None)
    if state is None:
        return None
    return BackgroundTasksView(
        queued=len(tasks.tasks),
        suppressed=state.suppressed_count,
        executed=state.executed,
        skipped=state.skipped,
    )
