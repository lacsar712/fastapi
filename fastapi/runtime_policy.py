"""Request-scoped runtime policy stack.

A cooperative-scope bound (``contextvars``) stack of policy frames that
constrains three independent runtime axes *during a request*:

* ``dep_admit`` -- whether new dependency values (including security
  dependencies and nested sub-dependencies) may be resolved;
* ``serialize_gate`` -- whether return values may be serialized into a
  response body (``jsonable_encoder`` and the JSON/ORJSON/UJSON response
  renders, including the Pydantic response serialization path);
* ``background_gate`` -- whether new background tasks may be enqueued, and
  whether tasks enqueued in the current segment must be observably cancelled
  when the segment pops.

The stack is intentionally decoupled from application/router permanent
defaults (default response class, ``dependency_overrides``), OpenAPI schema
generation, the Starlette middleware chain and static mounts.  When the
stack is empty every hook below is an exact pass-through to the pinned
behavior; no constructor gained a required parameter.

Frames are pushed/popped explicitly (``push_policy``/``pop_policy``) or via
the ``policy_frame`` context manager, which always pops -- including when the
body raises.  Nested sub-dependency callbacks and nested scopes read the
*current* stack top on every admission; they never reuse a snapshot taken
when an outer segment was entered.
"""

from __future__ import annotations

import contextlib
import functools
import itertools
import weakref
from collections.abc import Callable, Iterator
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from starlette.background import BackgroundTask

__all__ = [
    "BackgroundPolicy",
    "CancelledBackgroundTask",
    "DependencyResolutionDenied",
    "DepAdmitPolicy",
    "PolicyView",
    "RequestPolicyError",
    "RequestPolicyFrame",
    "BackgroundEnqueueDenied",
    "SerializationDenied",
    "SerializePolicy",
    "admit_background_enqueue",
    "admit_dependency",
    "admit_serialization",
    "background_task_is_cancelled",
    "get_policy_view",
    "policy_frame",
    "pop_policy",
    "push_policy",
    "register_background_task",
]


# ---------------------------------------------------------------------------
# Policy enums
# ---------------------------------------------------------------------------


class DepAdmitPolicy(str, Enum):
    """Admission policy for new dependency value resolution."""

    #: Align with the existing configuration: resolve dependencies as usual.
    HONOR_CFG = "honor_cfg"
    #: Reject every *new* dependency value taken in this frame observably.
    #: A dependency computation already in flight is never cancelled or
    #: rewritten into an empty success; only subsequent admissions are gated.
    DENY_NEW = "deny_new"
    #: Allow exactly one more successful dependency value admission in this
    #: frame; the next one is rejected.  Every admission re-reads the current
    #: stack top, so nested scopes never reuse an outer-segment snapshot.
    ONE_MORE = "one_more"


class SerializePolicy(str, Enum):
    """Gate policy for serializing a return value into response bytes."""

    #: Align with the existing configuration: serialize as usual.
    HONOR_CFG = "honor_cfg"
    #: Reject every *new* serialization path in this frame observably.  A
    #: render already in flight is not truncated or reported as a complete
    #: success; only subsequent serializations are gated.
    DENY_NEW = "deny_new"
    #: Every serialization requested in this frame must fail/refuse
    #: observably; it must never pretend final bytes were produced.  The
    #: caller's original return object is left untouched.
    FORCE_RAW = "force_raw"


class BackgroundPolicy(str, Enum):
    """Gate policy for background task enqueue/dispatch."""

    #: Align with the existing configuration: enqueue and run as usual.
    HONOR_CFG = "honor_cfg"
    #: Reject every *new* ``BackgroundTasks.add_task`` in this frame
    #: observably; the task must not be silently marked as enqueued.
    DENY_NEW = "deny_new"
    #: When this frame pops (normally or via exception), every task enqueued
    #: while the frame was active and still not executed is withdrawn to an
    #: observable cancellation.  Tasks belonging to other frames/requests are
    #: never touched.
    FORCE_CANCEL = "force_cancel"


# ---------------------------------------------------------------------------
# Observable error surfaces
# ---------------------------------------------------------------------------


class RequestPolicyError(RuntimeError):
    """Base class for observable rejections raised by the runtime policy stack."""

    #: Which axis rejected: ``"dep_admit"``, ``"serialize_gate"`` or
    #: ``"background_gate"``.
    axis: str = ""

    def __init__(self, message: str, *, frame_id: int, policy: str) -> None:
        super().__init__(message)
        #: Id of the frame that produced the rejection (correlation only).
        self.frame_id = frame_id
        #: Policy value active on the rejecting frame.
        self.policy = policy


class DependencyResolutionDenied(RequestPolicyError):
    """A new dependency value admission was rejected by the current stack."""

    axis = "dep_admit"

    def __init__(self, frame: RequestPolicyFrame, *, exhausted: bool = False) -> None:
        self.exhausted = exhausted
        message = (
            "Dependency admission denied by request runtime policy frame "
            f"#{frame.id}: dep_admit={frame.dep_admit.value!r}"
        )
        if exhausted:
            message += " (the single one_more admission was already consumed)"
        super().__init__(message, frame_id=frame.id, policy=frame.dep_admit.value)


class SerializationDenied(RequestPolicyError):
    """A new serialization/render path was rejected by the current stack."""

    axis = "serialize_gate"

    def __init__(self, frame: RequestPolicyFrame, *, kind: str) -> None:
        #: The serialization exit that was gated, e.g. ``"jsonable_encoder"``,
        #: ``"json_render"``, ``"orjson_render"``, ``"ujson_render"``,
        #: ``"response"``, ``"stream_item"`` or ``"sse_item"``.
        self.kind = kind
        super().__init__(
            f"Serialization ({kind!r}) denied by request runtime policy frame "
            f"#{frame.id}: serialize_gate={frame.serialize_gate.value!r}",
            frame_id=frame.id,
            policy=frame.serialize_gate.value,
        )


class BackgroundEnqueueDenied(RequestPolicyError):
    """A new background task enqueue was rejected by the current stack."""

    axis = "background_gate"

    def __init__(self, frame: RequestPolicyFrame) -> None:
        super().__init__(
            "Background task enqueue denied by request runtime policy frame "
            f"#{frame.id}: background_gate={frame.background_gate.value!r}",
            frame_id=frame.id,
            policy=frame.background_gate.value,
        )


@dataclass(frozen=True)
class CancelledBackgroundTask:
    """Observable record of a background task withdrawn before execution."""

    #: Frame that performed the cancellation.
    frame_id: int
    #: Qualified name of the task callable when available.
    func_name: str
    #: Short repr of the positional arguments.
    args_repr: str
    #: Short repr of the keyword arguments.
    kwargs_repr: str
    #: Why the task was withdrawn (``"force_cancel"``).
    reason: str


# ---------------------------------------------------------------------------
# Frame and stack
# ---------------------------------------------------------------------------


_frame_ids = itertools.count(1)


@dataclass
class RequestPolicyFrame:
    """A single frame on the request runtime policy stack.

    Instances are created by ``push_policy``/``policy_frame``; user code never
    constructs them directly.  Attributes are readable for observability.
    """

    dep_admit: DepAdmitPolicy
    serialize_gate: SerializePolicy
    background_gate: BackgroundPolicy
    on_background_cancel: Callable[[CancelledBackgroundTask], None] | None = None
    id: int = field(default_factory=lambda: next(_frame_ids))
    #: Remaining admissions for ``dep_admit=one_more``.
    remaining_dep_admissions: int = 1
    #: Tasks registered to this frame (enqueued while this frame was active).
    _enqueued: list[BackgroundTask] = field(default_factory=list)
    #: Observable cancellation records produced when this frame popped.
    _cancelled: list[CancelledBackgroundTask] = field(default_factory=list)
    #: ContextVar reset token captured at push.
    _token: Token[tuple[RequestPolicyFrame, ...]] | None = field(
        default=None, repr=False
    )

    @property
    def cancelled_tasks(self) -> tuple[CancelledBackgroundTask, ...]:
        """Records for tasks cancelled when this frame popped."""
        return tuple(self._cancelled)


#: The cooperative-scope bound stack.  Each asyncio/anyio task (and each
#: copied threadpool context) observes its own tuple; push/pop never mutates
#: tuples belonging to concurrent requests.
_policy_stack: ContextVar[tuple[RequestPolicyFrame, ...]] = ContextVar(
    "fastapi_request_runtime_policy_stack", default=()
)

# External execution ledgers.  They live in this module -- never as private
# attributes on Request/Response/APIRoute/BackgroundTasks/task objects.  Weak
# sets so finished tasks do not linger here.
_cancelled_tasks: weakref.WeakSet[Any] = weakref.WeakSet()
_running_tasks: weakref.WeakSet[Any] = weakref.WeakSet()
_completed_tasks: weakref.WeakSet[Any] = weakref.WeakSet()


@dataclass(frozen=True)
class PolicyView:
    """Merged view of the current stack combined with the pinned defaults.

    The values describe the *effective* policy: ``honor_cfg`` frames are
    transparent and the nearest non-honor frame per axis wins.  An empty
    stack reports ``honor_cfg`` on every axis, i.e. the existing behavior.
    """

    dep_admit: DepAdmitPolicy
    serialize_gate: SerializePolicy
    background_gate: BackgroundPolicy
    #: Remaining admissions when the effective dep policy is ``one_more``.
    dep_admit_remaining: int | None
    #: Number of frames currently pushed in this scope.
    depth: int


def get_policy_view() -> PolicyView:
    """Return the merged effective policy view for the current scope."""
    stack = _policy_stack.get()
    dep_admit = DepAdmitPolicy.HONOR_CFG
    serialize_gate = SerializePolicy.HONOR_CFG
    background_gate = BackgroundPolicy.HONOR_CFG
    dep_admit_remaining: int | None = None
    for frame in reversed(stack):
        if (
            dep_admit is DepAdmitPolicy.HONOR_CFG
            and frame.dep_admit is not DepAdmitPolicy.HONOR_CFG
        ):
            dep_admit = frame.dep_admit
            if dep_admit is DepAdmitPolicy.ONE_MORE:
                dep_admit_remaining = frame.remaining_dep_admissions
        if (
            serialize_gate is SerializePolicy.HONOR_CFG
            and frame.serialize_gate is not SerializePolicy.HONOR_CFG
        ):
            serialize_gate = frame.serialize_gate
        if (
            background_gate is BackgroundPolicy.HONOR_CFG
            and frame.background_gate is not BackgroundPolicy.HONOR_CFG
        ):
            background_gate = frame.background_gate
    return PolicyView(
        dep_admit=dep_admit,
        serialize_gate=serialize_gate,
        background_gate=background_gate,
        dep_admit_remaining=dep_admit_remaining,
        depth=len(stack),
    )


# ---------------------------------------------------------------------------
# Push / pop
# ---------------------------------------------------------------------------


def _coerce_enum(value: Any, enum_cls: type[Enum], name: str) -> Any:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:  # pragma: no cover - defensive
            pass
    raise TypeError(f"{name} must be a {enum_cls.__name__}, got {value!r}")


def push_policy(
    *,
    dep_admit: DepAdmitPolicy | str = DepAdmitPolicy.HONOR_CFG,
    serialize_gate: SerializePolicy | str = SerializePolicy.HONOR_CFG,
    background_gate: BackgroundPolicy | str = BackgroundPolicy.HONOR_CFG,
    on_background_cancel: Callable[[CancelledBackgroundTask], None] | None = None,
) -> RequestPolicyFrame:
    """Push a runtime policy frame onto the current cooperative scope.

    The frame stays active until ``pop_policy`` (or the ``policy_frame``
    context manager) unwinds it; inner pops restore outer frames.
    """
    frame = RequestPolicyFrame(
        dep_admit=_coerce_enum(dep_admit, DepAdmitPolicy, "dep_admit"),
        serialize_gate=_coerce_enum(serialize_gate, SerializePolicy, "serialize_gate"),
        background_gate=_coerce_enum(
            background_gate, BackgroundPolicy, "background_gate"
        ),
        on_background_cancel=on_background_cancel,
    )
    stack = _policy_stack.get()
    token = _policy_stack.set(stack + (frame,))
    frame._token = token
    return frame


def _cancel_pending_tasks(frame: RequestPolicyFrame) -> None:
    """Withdraw frame-ledger tasks that were enqueued but never executed."""
    for task in frame._enqueued:
        if task in _cancelled_tasks:
            continue
        if task in _running_tasks or task in _completed_tasks:
            # Already dispatched (possibly mid-flight): never abort an
            # in-flight task nor rewrite a completed one.
            continue
        _cancelled_tasks.add(task)
        func = getattr(task, "func", task)
        func_name = getattr(func, "__qualname__", None) or repr(func)
        record = CancelledBackgroundTask(
            frame_id=frame.id,
            func_name=func_name,
            args_repr=repr(getattr(task, "args", ()))[:500],
            kwargs_repr=repr(getattr(task, "kwargs", {}))[:500],
            reason="force_cancel",
        )
        frame._cancelled.append(record)
        if frame.on_background_cancel is not None:
            frame.on_background_cancel(record)


def pop_policy(frame: RequestPolicyFrame | None = None) -> RequestPolicyFrame:
    """Pop the current top frame (LIFO) and restore the previous view.

    A ``force_cancel`` frame withdraws its still-pending tasks at this point,
    whether the pop is normal or driven by an exception.
    """
    stack = _policy_stack.get()
    if not stack:
        raise RuntimeError("Cannot pop an empty request runtime policy stack")
    top = stack[-1]
    if frame is not None and frame is not top:
        raise RuntimeError(
            f"Cannot pop policy frame #{frame.id}: frame #{top.id} is on top"
        )
    token = top._token
    # Run cancellations while the frame is still the current top, so any
    # callback observing the scope sees the popping frame.
    if top.background_gate is BackgroundPolicy.FORCE_CANCEL:
        _cancel_pending_tasks(top)
    if token is not None:
        _policy_stack.reset(token)
    else:  # pragma: no cover - every pushed frame has a token
        _policy_stack.set(stack[:-1])
    return top


@contextlib.contextmanager
def policy_frame(
    *,
    dep_admit: DepAdmitPolicy | str = DepAdmitPolicy.HONOR_CFG,
    serialize_gate: SerializePolicy | str = SerializePolicy.HONOR_CFG,
    background_gate: BackgroundPolicy | str = BackgroundPolicy.HONOR_CFG,
    on_background_cancel: Callable[[CancelledBackgroundTask], None] | None = None,
) -> Iterator[RequestPolicyFrame]:
    """Context manager for one policy segment; pops cleanly even on raise."""
    frame = push_policy(
        dep_admit=dep_admit,
        serialize_gate=serialize_gate,
        background_gate=background_gate,
        on_background_cancel=on_background_cancel,
    )
    try:
        yield frame
    finally:
        # Pop unconditionally: a raised body must leave the scope clean and a
        # force_cancel frame still withdraws its pending tasks.
        pop_policy(frame)


# ---------------------------------------------------------------------------
# Admission hooks (called from the wired exit points)
# ---------------------------------------------------------------------------


def admit_dependency() -> None:
    """Gate one new dependency value admission at the *current* stack top.

    Honors re-read the stack and defer to outer frames; ``one_more`` consumes
    one admission from the owning frame; ``deny_new`` raises observably.  A
    dependency whose computation already started is never affected: this hook
    is only consulted *before* a new subtree is entered.
    """
    for frame in reversed(_policy_stack.get()):
        policy = frame.dep_admit
        if policy is DepAdmitPolicy.HONOR_CFG:
            continue
        if policy is DepAdmitPolicy.DENY_NEW:
            raise DependencyResolutionDenied(frame)
        # one_more
        if frame.remaining_dep_admissions > 0:
            frame.remaining_dep_admissions -= 1
            return
        raise DependencyResolutionDenied(frame, exhausted=True)


def admit_serialization(kind: str) -> None:
    """Gate one new serialization exit at the *current* stack top.

    ``kind`` identifies the exit (response serialization, jsonable encoder,
    one JSON render, one streamed item).  The hook is consulted at entry only,
    so a render already in flight is never truncated.
    """
    for frame in reversed(_policy_stack.get()):
        policy = frame.serialize_gate
        if policy is SerializePolicy.HONOR_CFG:
            continue
        raise SerializationDenied(frame, kind=kind)


def admit_background_enqueue() -> None:
    """Gate one new ``BackgroundTasks.add_task`` at the *current* stack top.

    ``force_cancel`` does not reject enqueues -- the tasks are admitted and
    withdrawn observably when the frame pops.
    """
    for frame in reversed(_policy_stack.get()):
        if frame.background_gate is BackgroundPolicy.DENY_NEW:
            raise BackgroundEnqueueDenied(frame)


def register_background_task(task: BackgroundTask) -> None:
    """Attribute an admitted task to every currently active frame.

    The ledger lives in the frames themselves -- nothing is attached to the
    task or to ``BackgroundTasks``.
    """
    stack = _policy_stack.get()
    for frame in stack:
        frame._enqueued.append(task)


def background_task_is_cancelled(task: BackgroundTask) -> bool:
    """Whether a task was withdrawn before its dispatch."""
    return task in _cancelled_tasks


# ---------------------------------------------------------------------------
# Patched exit points on the Starlette classes FastAPI builds on.
#
# The wrappers are inert when the stack is empty (direct call-through with
# identical inputs/outputs), so pinned behavior is unchanged unless a frame
# is explicitly pushed.  The middleware kernel, Request/Response/Route
# classes and the BackgroundTasks containers are not otherwise modified.
# ---------------------------------------------------------------------------

_patches_installed = False


def _install_patches() -> None:
    global _patches_installed
    if _patches_installed:
        return

    from starlette.background import BackgroundTask, BackgroundTasks
    from starlette.responses import JSONResponse

    original_add_task = BackgroundTasks.add_task
    original_task_call = BackgroundTask.__call__
    original_json_render = JSONResponse.render

    @functools.wraps(original_add_task)
    def _gated_add_task(
        self: BackgroundTasks, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        # Reject before appending: a denied task must never be silently
        # marked as enqueued.
        admit_background_enqueue()
        original_add_task(self, func, *args, **kwargs)
        register_background_task(self.tasks[-1])

    async def _gated_task_call(self: BackgroundTask) -> None:
        if background_task_is_cancelled(self):
            # Withdrawn by a popped force_cancel frame; the cancellation was
            # recorded at pop time.
            return
        _running_tasks.add(self)
        try:
            await original_task_call(self)
        finally:
            _running_tasks.discard(self)
            _completed_tasks.add(self)

    @functools.wraps(original_json_render)
    def _gated_json_render(self: JSONResponse, content: Any) -> bytes:
        admit_serialization("json_render")
        return original_json_render(self, content)

    # Method replacement on Starlette classes; inert pass-through while the
    # policy stack is empty.  mypy calls these method-assign and ty
    # invalid-assignment; the middle assignment only trips mypy, hence the
    # different ignore granularity.
    BackgroundTasks.add_task = _gated_add_task  # type: ignore
    BackgroundTask.__call__ = _gated_task_call  # type: ignore[method-assign]
    JSONResponse.render = _gated_json_render  # type: ignore
    _patches_installed = True


_install_patches()
