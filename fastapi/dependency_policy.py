"""Request-scoped dependency solving and background task policy stack.

The policy stack lives in the ASGI ``scope`` of the current HTTP request (or
the WebSocket connection after the handshake), never in process level state.
Two concurrent requests therefore push onto independent stacks and can never
observe each other's frames.

Each frame carries four independent policy dimensions:

* :class:`SolveReuse` -- how ``Dependant.use_cache``/``dependency_cache`` reuse
  behaves while solving dependencies in this frame.
* :class:`OverrideLatch` -- how ``dependency_overrides`` mappings are consulted
  while solving dependencies in this frame.
* :class:`TaskAdmit` -- how ``BackgroundTasks.add_task`` admission and the
  post-response execution face behave in this frame.
* :class:`AstackRelease` -- when generator/async-generator dependencies entered
  in this frame are closed.

An empty stack is exactly equivalent to the historical FastAPI behavior.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from starlette.types import ASGIApp, Receive, Scope, Send

if TYPE_CHECKING:
    from fastapi.background import BackgroundTasks
    from fastapi.types import DependencyCacheKey

POLICY_STACK_SCOPE_KEY = "fastapi_dependency_policy_stack"


class SolveReuse(str, Enum):
    """Effective reuse policy for ``Dependant.use_cache``."""

    HONOR_REUSE = "honor_reuse"
    FORCE_REINVOKE = "force_reinvoke"
    PIN_CACHED = "pin_cached"


class OverrideLatch(str, Enum):
    """Effective latch policy for ``dependency_overrides``."""

    HONOR_MAP = "honor_map"
    SKIP_MAP = "skip_map"
    FREEZE_AT_ENTER = "freeze_at_enter"


class TaskAdmit(str, Enum):
    """Effective admission policy for ``BackgroundTasks.add_task``."""

    HONOR_ENQUEUE = "honor_enqueue"
    REFUSE_ENQUEUE = "refuse_enqueue"
    MUTE_AFTER_RESPONSE = "mute_after_response"


class AstackRelease(str, Enum):
    """Effective release policy for generator dependency exit stacks."""

    HONOR_RESPONSE_EXIT = "honor_response_exit"
    DEFER_UNTIL_POP = "defer_until_pop"
    CLOSE_ON_POP = "close_on_pop"


class DependencyPolicyError(RuntimeError):
    """Raised when a policy frame cannot be pushed or popped as requested."""


class TaskRefusedError(DependencyPolicyError):
    """Raised when ``BackgroundTasks.add_task`` is refused by an active frame."""


@dataclass
class _PolicyFrame:
    solve_reuse: SolveReuse
    override_latch: OverrideLatch
    task_admit: TaskAdmit
    astack_release: AstackRelease
    frozen_overrides: Mapping[Callable[..., Any], Callable[..., Any]] | None
    pinned_keys: set[DependencyCacheKey] = field(default_factory=set)
    registered_generators: int = 0
    _function_astack: AsyncExitStack | None = None
    _request_astack: AsyncExitStack | None = None

    def astack_for(self, kind: str) -> AsyncExitStack:
        if kind == "function":
            if self._function_astack is None:
                self._function_astack = AsyncExitStack()
            return self._function_astack
        if self._request_astack is None:
            self._request_astack = AsyncExitStack()
        return self._request_astack

    async def aclose(self) -> None:
        # Function scoped generators historically finalize before request
        # scoped ones, keep that order when closing a frame.
        for owned_stack in (self._function_astack, self._request_astack):
            if owned_stack is not None:
                await owned_stack.aclose()


@dataclass
class _PolicyStackState:
    scope: Scope
    frames: list[_PolicyFrame] = field(default_factory=list)
    admitted_tasks: int = 0
    entered_generators: int = 0
    queues: list[BackgroundTasks] = field(default_factory=list)

    def top_frame(self) -> _PolicyFrame | None:
        return self.frames[-1] if self.frames else None


@dataclass(frozen=True)
class EffectiveDependencyPolicy:
    """Read-only effective view for the current request/connection.

    The values are merged with the live ``Dependant.use_cache`` flags, the
    current ``dependency_overrides`` mapping, the request bound
    ``BackgroundTasks`` queues and the generator exit stack bookkeeping.
    """

    depth: int
    solve_reuse: SolveReuse
    override_latch: OverrideLatch
    task_admit: TaskAdmit
    astack_release: AstackRelease
    dependency_overrides: Mapping[Callable[..., Any], Callable[..., Any]]
    admitted_tasks: int
    queued_tasks: int
    muted_tasks: int
    registered_generators: int

    def cache_reuse_allowed(self, dependant_use_cache: bool) -> bool:
        """Effective reuse after combining ``Dependant.use_cache`` with frame."""
        if self.solve_reuse is SolveReuse.FORCE_REINVOKE:
            return False
        return dependant_use_cache


def _resolve_scope(connection_or_scope: Any) -> Scope:
    scope = getattr(connection_or_scope, "scope", connection_or_scope)
    if not isinstance(scope, dict):
        raise TypeError(
            "Expected a Request, WebSocket, HTTPConnection or ASGI scope mapping,"
            f" got: {type(connection_or_scope)!r}"
        )
    return scope


def _ensure_policy_state(scope: Scope) -> _PolicyStackState:
    state = scope.get(POLICY_STACK_SCOPE_KEY)
    if state is None:
        state = _PolicyStackState(scope=scope)
        scope[POLICY_STACK_SCOPE_KEY] = state
    return state


def _get_policy_state(scope: Scope) -> _PolicyStackState | None:
    state = scope.get(POLICY_STACK_SCOPE_KEY)
    if state is not None:
        assert isinstance(state, _PolicyStackState)
    return state


def _top_frame(scope: Scope) -> tuple[_PolicyStackState | None, _PolicyFrame | None]:
    state = _get_policy_state(scope)
    if state is None:
        return None, None
    return state, state.top_frame()


def push_dependency_policy(
    connection_or_scope: Any,
    *,
    solve_reuse: SolveReuse = SolveReuse.HONOR_REUSE,
    override_latch: OverrideLatch = OverrideLatch.HONOR_MAP,
    task_admit: TaskAdmit = TaskAdmit.HONOR_ENQUEUE,
    astack_release: AstackRelease = AstackRelease.HONOR_RESPONSE_EXIT,
    dependency_overrides_provider: Any | None = None,
) -> _PolicyFrame:
    """Push a policy frame onto the current request/connection stack.

    The frame takes effect immediately for any dependency solving,
    ``add_task`` admission or generator entry performed afterwards; nothing is
    snapshotted from a "solve start" moment. Pushing a frame that could not
    take effect for already-done work raises :class:`DependencyPolicyError`
    instead of silently pretending it applied retroactively.
    """
    scope = _resolve_scope(connection_or_scope)
    state = _ensure_policy_state(scope)
    if task_admit is TaskAdmit.REFUSE_ENQUEUE and state.admitted_tasks:
        raise DependencyPolicyError(
            "Cannot push a 'refuse_enqueue' frame after background tasks have"
            " already been admitted to this request; already queued tasks must"
            " not be cleared by the refusal gate"
        )
    if astack_release is AstackRelease.CLOSE_ON_POP and state.entered_generators:
        raise DependencyPolicyError(
            "Cannot push a 'close_on_pop' frame after generator dependencies"
            " have already been entered; their exit stacks cannot be moved"
            " retroactively"
        )
    frozen_overrides: Mapping[Callable[..., Any], Callable[..., Any]] | None = None
    if override_latch is OverrideLatch.FREEZE_AT_ENTER:
        provider = dependency_overrides_provider
        if provider is None:
            provider = scope.get("app")
        mapping = getattr(provider, "dependency_overrides", None)
        frozen_overrides = MappingProxyType(dict(mapping or {}))
    frame = _PolicyFrame(
        solve_reuse=solve_reuse,
        override_latch=override_latch,
        task_admit=task_admit,
        astack_release=astack_release,
        frozen_overrides=frozen_overrides,
    )
    state.frames.append(frame)
    return frame


async def pop_dependency_policy(
    connection_or_scope: Any, frame: _PolicyFrame | None = None
) -> None:
    """Pop a frame (default: the top frame), closing the generators it owns.

    Frames pushed above ``frame`` are popped first, in order, so the stack can
    always be emptied cleanly, including on exception paths.
    """
    scope = _resolve_scope(connection_or_scope)
    state = _get_policy_state(scope)
    if state is None or not state.frames:
        raise DependencyPolicyError(
            "Cannot pop a dependency policy frame: stack is empty"
        )
    if frame is None:
        target = state.frames[-1]
    else:
        if frame not in state.frames:
            raise DependencyPolicyError("Cannot pop a frame that is not on this stack")
        target = frame
    index = state.frames.index(target)
    to_close = [state.frames[i] for i in range(len(state.frames) - 1, index - 1, -1)]
    del state.frames[index:]
    # Always unwind every frame so one failing generator cleanup cannot leave
    # sibling frames (and their generators) unclosed on exception paths.
    first_error: BaseException | None = None
    for current in to_close:
        try:
            await current.aclose()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


@asynccontextmanager
async def dependency_policy(
    connection_or_scope: Any,
    *,
    solve_reuse: SolveReuse = SolveReuse.HONOR_REUSE,
    override_latch: OverrideLatch = OverrideLatch.HONOR_MAP,
    task_admit: TaskAdmit = TaskAdmit.HONOR_ENQUEUE,
    astack_release: AstackRelease = AstackRelease.HONOR_RESPONSE_EXIT,
    dependency_overrides_provider: Any | None = None,
) -> AsyncIterator[_PolicyFrame]:
    """Async context manager pushing a frame and guaranteeing a clean pop."""
    frame = push_dependency_policy(
        connection_or_scope,
        solve_reuse=solve_reuse,
        override_latch=override_latch,
        task_admit=task_admit,
        astack_release=astack_release,
        dependency_overrides_provider=dependency_overrides_provider,
    )
    try:
        yield frame
    finally:
        await pop_dependency_policy(connection_or_scope, frame)


def _live_overrides(
    dependency_overrides_provider: Any | None,
) -> Mapping[Callable[..., Any], Callable[..., Any]]:
    if dependency_overrides_provider is not None:
        mapping = getattr(dependency_overrides_provider, "dependency_overrides", None)
        if mapping:
            assert isinstance(mapping, Mapping)
            return mapping
    return {}


def effective_overrides_mapping(
    scope: Scope, dependency_overrides_provider: Any | None
) -> Mapping[Callable[..., Any], Callable[..., Any]]:
    """Effective ``dependency_overrides`` mapping for solving in this scope."""
    state, frame = _top_frame(scope)
    if state is None or frame is None:
        return _live_overrides(dependency_overrides_provider)
    if frame.override_latch is OverrideLatch.SKIP_MAP:
        return {}
    if frame.override_latch is OverrideLatch.FREEZE_AT_ENTER:
        return frame.frozen_overrides or {}
    return _live_overrides(dependency_overrides_provider)


def cache_lookup(
    scope: Scope,
    cache: dict[DependencyCacheKey, Any],
    key: DependencyCacheKey,
    *,
    dependant_use_cache: bool,
) -> tuple[bool, Any]:
    """Return ``(hit, value)`` for a dependency cache key under current policy."""
    if not dependant_use_cache:
        return False, None
    _, frame = _top_frame(scope)
    if frame is not None and frame.solve_reuse is SolveReuse.FORCE_REINVOKE:
        return False, None
    if key in cache:
        if frame is not None and frame.solve_reuse is SolveReuse.PIN_CACHED:
            frame.pinned_keys.add(key)
        return True, cache[key]
    return False, None


def _any_frame_pins(state: _PolicyStackState, key: DependencyCacheKey) -> bool:
    return any(key in frame.pinned_keys for frame in state.frames)


def cache_store(
    scope: Scope,
    cache: dict[DependencyCacheKey, Any],
    key: DependencyCacheKey,
    value: Any,
    *,
    dependant_use_cache: bool,
) -> None:
    """Write a solved value honoring ``force_reinvoke``/``pin_cached`` frames."""
    state, frame = _top_frame(scope)
    if frame is not None and frame.solve_reuse is SolveReuse.PIN_CACHED:
        frame.pinned_keys.add(key)
    pinned = state is not None and _any_frame_pins(state, key)
    if key in cache:
        if (
            not pinned
            and dependant_use_cache
            and frame is not None
            and frame.solve_reuse is SolveReuse.FORCE_REINVOKE
        ):
            cache[key] = value
        return
    cache[key] = value


def generator_astack(
    scope: Scope,
    *,
    is_function_scope: bool,
    request_astack: AsyncExitStack,
    function_astack: AsyncExitStack,
) -> AsyncExitStack:
    """Pick the exit stack a generator dependency must be entered into."""
    # Track every generator entry, even before any frame is pushed, so a later
    # 'close_on_pop' push can be rejected instead of claiming generators whose
    # exit stacks were already registered.
    state = _ensure_policy_state(scope)
    frame = state.top_frame()
    state.entered_generators += 1
    if frame is None or frame.astack_release is AstackRelease.HONOR_RESPONSE_EXIT:
        return function_astack if is_function_scope else request_astack
    frame.registered_generators += 1
    kind = "function" if is_function_scope else "request"
    return frame.astack_for(kind)


def bind_background_tasks(scope: Scope, tasks: BackgroundTasks) -> None:
    """Bind a request-injected ``BackgroundTasks`` instance to this scope."""
    tasks._fastapi_bind_policy_scope(scope)
    state = _ensure_policy_state(scope)
    state.queues.append(tasks)


def get_effective_dependency_policy(
    connection_or_scope: Any,
) -> EffectiveDependencyPolicy:
    """Return the read-only merged effective policy view for this request."""
    scope = _resolve_scope(connection_or_scope)
    state = _get_policy_state(scope)
    app = scope.get("app")
    live_mapping = getattr(app, "dependency_overrides", None) or {}
    if state is None or not state.frames:
        return EffectiveDependencyPolicy(
            depth=0,
            solve_reuse=SolveReuse.HONOR_REUSE,
            override_latch=OverrideLatch.HONOR_MAP,
            task_admit=TaskAdmit.HONOR_ENQUEUE,
            astack_release=AstackRelease.HONOR_RESPONSE_EXIT,
            dependency_overrides=MappingProxyType(dict(live_mapping)),
            admitted_tasks=0,
            queued_tasks=0,
            muted_tasks=0,
            registered_generators=0,
        )
    frame = state.top_frame()
    assert frame is not None
    if frame.override_latch is OverrideLatch.SKIP_MAP:
        mapping: Mapping[Callable[..., Any], Callable[..., Any]] = {}
    elif frame.override_latch is OverrideLatch.FREEZE_AT_ENTER:
        mapping = frame.frozen_overrides or {}
    else:
        mapping = MappingProxyType(dict(live_mapping))
    queued = sum(len(queue.tasks) for queue in state.queues)
    muted = sum(len(queue._fastapi_policy_muted()) for queue in state.queues)
    return EffectiveDependencyPolicy(
        depth=len(state.frames),
        solve_reuse=frame.solve_reuse,
        override_latch=frame.override_latch,
        task_admit=frame.task_admit,
        astack_release=frame.astack_release,
        dependency_overrides=mapping,
        admitted_tasks=state.admitted_tasks,
        queued_tasks=queued,
        muted_tasks=muted,
        registered_generators=state.entered_generators,
    )


async def _finalize_policy_stack(scope: Scope) -> None:
    state = _get_policy_state(scope)
    if state is None:
        return
    while state.frames:
        await pop_dependency_policy(scope)


class DependencyPolicyFinalizeMiddleware:
    """Outermost safety net: pop frames users forgot (also on exceptions).

    It runs after any user middleware code following ``await app(...)``, so a
    frame explicitly popped there (e.g. ``defer_until_pop`` after the response)
    is left untouched; only genuinely remaining frames are finalized.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await self.app(scope, receive, send)
        finally:
            if scope["type"] in ("http", "websocket"):
                await _finalize_policy_stack(scope)
