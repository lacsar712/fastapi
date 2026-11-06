"""Acceptance tests for the request-bound dependency/background policy stack.

Covers the four independent dimensions (cache_surface, override_surface,
bg_admit, gen_exit), the read-only effective view, per-request binding,
concurrent isolation, late-push rules, WebSocket behavior and empty-stack
parity with plain FastAPI.
"""

import asyncio
import threading
from contextlib import AsyncExitStack
from typing import Any

import httpx
import pytest
from fastapi import (
    BackgroundAdmit,
    BackgroundTaskRejectedError,
    BackgroundTasks,
    CacheSurface,
    Depends,
    FastAPI,
    GeneratorExitPolicy,
    OverrideSurface,
    PolicyFrame,
    Request,
    RequestPolicyStackError,
    Security,
    WebSocket,
    get_background_tasks_view,
    get_frozen_cache_keys,
    get_policy_view,
    pop_request_policy,
    push_request_policy,
    request_policy,
)
from fastapi.dependencies.utils import get_dependant, solve_dependencies
from fastapi.request_policy import _PolicyCache, ensure_request_policy_stack
from fastapi.security import SecurityScopes
from fastapi.testclient import TestClient

_lock = threading.Lock()


def _reset_calls() -> None:
    calls["value"] = 0
    bg_ran.clear()
    gen_events.clear()


calls = {"value": 0}
bg_ran: list[str] = []
gen_events: list[str] = []


# ---------------------------------------------------------------------------
# Shared dependencies
# ---------------------------------------------------------------------------


async def dep_value() -> int:
    calls["value"] += 1
    return calls["value"]


async def dep_parent(value: int = Depends(dep_value)) -> int:
    return value


def get_data() -> str:
    return "original-data"


def get_data_override() -> str:
    return "override-data"


def get_data_changed() -> str:
    return "changed-after-push"


def get_user(security_scopes: SecurityScopes) -> str:
    return f"original-user:{security_scopes.scopes}"


def get_user_override(security_scopes: SecurityScopes) -> str:
    return f"override-user:{security_scopes.scopes}"


async def gen_dependency() -> Any:
    gen_events.append("enter")
    try:
        yield "gen-value"
    finally:
        gen_events.append("exit")


# ---------------------------------------------------------------------------
# App 1: cache surface (mid-solve push demonstrates bypass vs honor)
# ---------------------------------------------------------------------------

cache_app = FastAPI()


async def dep_push_cache(request: Request) -> PolicyFrame:
    surface = request.headers.get("x-cache-surface", "honor")
    return push_request_policy(
        request,
        cache_surface=(
            CacheSurface.bypass_cache
            if surface == "bypass"
            else CacheSurface.honor_cache
        ),
    )


@cache_app.get("/cache-mid")
async def cache_mid(
    first: int = Depends(dep_value),
    frame: PolicyFrame = Depends(dep_push_cache),
    parent: int = Depends(dep_parent),
    again: int = Depends(dep_value),
) -> dict[str, int]:
    # The frame is popped inside the request, before the response is sent.
    await pop_request_policy(frame)
    return {
        "first": first,
        "parent": parent,
        "again": again,
        "calls": calls["value"],
    }


# ---------------------------------------------------------------------------
# App 2: override surface (ignore / freeze) incl. Security dependencies
# ---------------------------------------------------------------------------

override_app = FastAPI()


@override_app.get("/whoami")
def whoami(
    user: str = Security(get_user, scopes=["read"]),
    data: str = Depends(get_data),
) -> dict[str, str]:
    return {"user": user, "data": data}


class OverrideSurfaceMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive, send)
        mode = request.headers.get("x-override-surface", "honor")
        if mode == "ignore":
            async with request_policy(
                request, override_surface=OverrideSurface.ignore_overrides
            ):
                await self.app(scope, receive, send)
        elif mode == "freeze":
            frame = push_request_policy(
                request, override_surface=OverrideSurface.freeze_map
            )
            try:
                # Mutate AFTER push: the frame must keep its entry snapshot.
                override_app.dependency_overrides[get_data] = get_data_changed
                del override_app.dependency_overrides[get_user]
                await self.app(scope, receive, send)
            finally:
                # Restore the mapping, then pop -- the frame never saw the
                # mutation; this is not relying on mutate-and-restore for
                # correctness, it is pure test cleanup.
                override_app.dependency_overrides[get_data] = get_data_override
                override_app.dependency_overrides[get_user] = get_user_override
                await pop_request_policy(frame)
        else:
            await self.app(scope, receive, send)


override_app.add_middleware(OverrideSurfaceMiddleware)


# ---------------------------------------------------------------------------
# App 3: background task admit (reject / suppress / honor) + gen_exit
# ---------------------------------------------------------------------------

policy_app = FastAPI()


def record_bg(name: str) -> None:
    with _lock:
        bg_ran.append(name)


@policy_app.get("/bg-by-middleware")
async def bg_by_middleware(background_tasks: BackgroundTasks) -> dict[str, Any]:
    rejected = False
    try:
        background_tasks.add_task(record_bg, "late")
    except BackgroundTaskRejectedError:
        rejected = True
    view = get_background_tasks_view(background_tasks)
    return {
        "rejected": rejected,
        "queued": view.queued if view is not None else len(background_tasks.tasks),
        "suppressed": view.suppressed if view is not None else 0,
    }


async def dep_enqueue_early(
    request: Request, background_tasks: BackgroundTasks
) -> PolicyFrame:
    # Enqueued BEFORE the gate exists: reject must never purge it.
    background_tasks.add_task(record_bg, "early")
    return push_request_policy(request, bg_admit=BackgroundAdmit.reject_add)


@policy_app.get("/bg-reject-late-push")
async def bg_reject_late_push(
    background_tasks: BackgroundTasks,
    frame: PolicyFrame = Depends(dep_enqueue_early),
) -> dict[str, Any]:
    rejected = False
    try:
        background_tasks.add_task(record_bg, "late")
    except BackgroundTaskRejectedError:
        rejected = True
    await pop_request_policy(frame)
    return {"rejected": rejected, "queued": len(background_tasks.tasks)}


@policy_app.get("/view")
async def view_layers(request: Request) -> dict[str, Any]:
    v0 = get_policy_view(request)
    f1 = push_request_policy(
        request,
        cache_surface=CacheSurface.bypass_cache,
        bg_admit=BackgroundAdmit.reject_add,
    )
    v1 = get_policy_view(request)
    f2 = push_request_policy(request, bg_admit=BackgroundAdmit.suppress_run)
    v2 = get_policy_view(request)
    await pop_request_policy(f2)
    v3 = get_policy_view(request)
    await pop_request_policy(f1)
    v4 = get_policy_view(request)
    # Frames must be popped in LIFO order.
    outer = push_request_policy(request, bg_admit=BackgroundAdmit.reject_add)
    inner = push_request_policy(request, bg_admit=BackgroundAdmit.suppress_run)
    bad_pop = False
    try:
        await pop_request_policy(outer)
    except RequestPolicyStackError:
        bad_pop = True
    # Correct cleanup: inner first, then outer.
    await pop_request_policy(inner)
    await pop_request_policy(outer)
    return {
        "v0": [
            v0.cache_surface.value,
            v0.override_surface.value,
            v0.bg_admit.value,
            v0.gen_exit.value,
            v0.depth,
        ],
        "v1": [v1.bg_admit.value, v1.cache_surface.value, v1.depth],
        "v2": [v2.bg_admit.value, v2.cache_surface.value, v2.depth],
        "v3": [v3.bg_admit.value, v3.cache_surface.value, v3.depth],
        "v4": [v4.bg_admit.value, v4.depth],
        "bad_pop": bad_pop,
    }


class BackgroundGenMiddleware:
    def __init__(self, app: Any, mode_header: str) -> None:
        self.app = app
        self.mode_header = mode_header

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive, send)
        bg_mode = request.headers.get("x-bg")
        gen_mode = request.headers.get(self.mode_header)
        kwargs: dict[str, Any] = {}
        if bg_mode == "reject":
            kwargs["bg_admit"] = BackgroundAdmit.reject_add
        elif bg_mode == "suppress":
            kwargs["bg_admit"] = BackgroundAdmit.suppress_run
        if gen_mode == "hold":
            kwargs["gen_exit"] = GeneratorExitPolicy.hold_until_pop
        elif gen_mode == "force":
            kwargs["gen_exit"] = GeneratorExitPolicy.force_close_now
        frame = push_request_policy(request, **kwargs) if kwargs else None
        try:
            await self.app(scope, receive, send)
            gen_events.append("after-response")
        finally:
            if frame is not None:
                await pop_request_policy(frame)
        gen_events.append("after-pop")


policy_app.add_middleware(BackgroundGenMiddleware, mode_header="x-gen")


@policy_app.get("/gen-honor")
async def gen_honor(value: str = Depends(gen_dependency)) -> dict[str, Any]:
    return {"closed_in_endpoint": "exit" in gen_events, "value": value}


async def dep_push_force(request: Request) -> PolicyFrame:
    return push_request_policy(request, gen_exit=GeneratorExitPolicy.force_close_now)


@policy_app.get("/gen-force")
async def gen_force(
    frame: PolicyFrame = Depends(dep_push_force),
    value: str = Depends(gen_dependency),
) -> dict[str, Any]:
    # Frame pushed BEFORE this generator dependency was solved: pop closes
    # it immediately, while the response has not finished yet.
    await pop_request_policy(frame)
    return {"closed_at_pop": "exit" in gen_events, "value": value}


@policy_app.get("/gen-late-push")
async def gen_late_push(
    value: str = Depends(gen_dependency),
    frame: PolicyFrame = Depends(dep_push_force),
) -> dict[str, Any]:
    # The generator was already entered on the normal astack when the
    # force_close_now frame is pushed. Popping must NOT capture it
    # retroactively; it still exits at response end.
    await pop_request_policy(frame)
    return {"closed_at_pop": "exit" in gen_events, "value": value}


@policy_app.get("/gen-hold-error")
async def gen_hold_error(value: str = Depends(gen_dependency)) -> dict[str, Any]:
    raise RuntimeError("boom")


@policy_app.get("/plain")
async def plain(value: int = Depends(dep_value)) -> dict[str, Any]:
    view = get_policy_view()
    return {"value": value, "default_bg": view.bg_admit.value}


@policy_app.get("/bg-contextvar")
async def bg_contextvar(background_tasks: BackgroundTasks) -> dict[str, Any]:
    # Push without a Request/WebSocket handle: the request-bound stack is
    # found via context propagation (nested call support).
    frame = push_request_policy(bg_admit=BackgroundAdmit.reject_add)
    rejected = False
    try:
        background_tasks.add_task(record_bg, "ctx")
    except BackgroundTaskRejectedError:
        rejected = True
    finally:
        await pop_request_policy(frame)
    return {"rejected": rejected}


# ---------------------------------------------------------------------------
# App 4: WebSocket override fork + bg reject
# ---------------------------------------------------------------------------

ws_app = FastAPI()


def ws_dep() -> str:
    return "ws-original"


def ws_dep_override() -> str:
    return "ws-override"


@ws_app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket, data: str = Depends(ws_dep)) -> None:
    await websocket.accept()
    await websocket.send_text(data)


@ws_app.websocket("/ws-bg")
async def ws_background(websocket: WebSocket, tasks: BackgroundTasks) -> None:
    await websocket.accept()
    tasks.add_task(record_bg, "ws-early")
    frame = push_request_policy(websocket, bg_admit=BackgroundAdmit.reject_add)
    rejected = False
    try:
        tasks.add_task(record_bg, "ws-late")
    except BackgroundTaskRejectedError:
        rejected = True
    await pop_request_policy(frame)
    await websocket.send_json({"rejected": rejected, "queued": len(tasks.tasks)})


class WsOverrideMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "websocket":
            await self.app(scope, receive, send)
            return
        mode = None
        for key, value in scope.get("headers", []):
            if key == b"x-ws-override":
                mode = value
                break
        if mode == b"ignore":
            connection = WebSocket(scope, receive, send)
            async with request_policy(
                connection, override_surface=OverrideSurface.ignore_overrides
            ):
                await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)


ws_app.add_middleware(WsOverrideMiddleware)


# ---------------------------------------------------------------------------
# App 5: concurrency isolation (different bg_admit per concurrent request)
# ---------------------------------------------------------------------------

concurrent_app = FastAPI()

barrier_events: dict[str, asyncio.Event] = {}


class BgHeaderMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive, send)
        bg_mode = request.headers.get("x-bg", "honor")
        admit = (
            BackgroundAdmit.reject_add
            if bg_mode == "reject"
            else BackgroundAdmit.honor_queue
        )
        async with request_policy(request, bg_admit=admit):
            await self.app(scope, receive, send)


concurrent_app.add_middleware(BgHeaderMiddleware)


@concurrent_app.get("/concurrent")
async def concurrent(
    request: Request, background_tasks: BackgroundTasks
) -> dict[str, Any]:
    tag = request.headers["x-tag"]
    event = asyncio.Event()
    barrier_events[tag] = event
    await asyncio.wait_for(_wait_partner(tag), timeout=5)
    admitted = True
    try:
        background_tasks.add_task(record_bg, tag)
    except BackgroundTaskRejectedError:
        admitted = False
    return {"admitted": admitted, "token": get_policy_view(request).stack_token}


async def _wait_partner(tag: str) -> None:
    other = "b" if tag == "a" else "a"
    for _ in range(100):
        if other in barrier_events:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("partner request never arrived")


# ===========================================================================
# Tests
# ===========================================================================


def test_cache_honor_vs_bypass_on_subdependency() -> None:
    client = TestClient(cache_app)

    _reset_calls()
    response = client.get("/cache-mid", headers={"x-cache-surface": "honor"})
    assert response.status_code == 200
    # honor: the use_cache sub-dependency is solved once and reused everywhere.
    assert response.json() == {
        "first": 1,
        "parent": 1,
        "again": 1,
        "calls": 1,
    }

    _reset_calls()
    response = client.get("/cache-mid", headers={"x-cache-surface": "bypass"})
    assert response.status_code == 200
    data = response.json()
    # bypass pushed mid-recursion: the pre-frame cached value is ignored by
    # the later parent sub-dependency (re-invoked), while the value freshly
    # solved inside the frame is still reused by `again`.
    assert data["first"] == 1
    assert data["parent"] == 2
    assert data["again"] == 2
    assert data["calls"] == 2


def test_freeze_solved_blocks_foreign_overwrite_but_allows_owner_reads() -> None:
    async def scenario() -> tuple[Any, ...]:
        dep = get_dependant(path="/", call=_freeze_endpoint)
        cache: Any = _PolicyCache()

        def scope_for() -> dict[str, Any]:
            return {
                "type": "http",
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/",
                "raw_path": b"/",
                "query_string": b"",
                "headers": [],
                "server": ("test", 80),
                "client": None,
            }

        async with AsyncExitStack() as stack:
            # Request A pushes freeze_solved and solves first.
            scope_a = scope_for()
            scope_a["fastapi_inner_astack"] = stack
            scope_a["fastapi_function_astack"] = stack
            rp_a = ensure_request_policy_stack(scope_a)
            push_request_policy(
                Request(scope_a), cache_surface=CacheSurface.freeze_solved
            )
            result_a = await solve_dependencies(
                request=Request(scope_a),
                dependant=dep,
                dependency_cache=cache,
                async_exit_stack=stack,
                embed_body_fields=False,
            )

            # A concurrent request B with its own stack bypasses the shared
            # cache and recomputes: it must not overwrite/clear the frozen key.
            scope_b = scope_for()
            scope_b["fastapi_inner_astack"] = stack
            scope_b["fastapi_function_astack"] = stack
            ensure_request_policy_stack(scope_b)
            push_request_policy(
                Request(scope_b), cache_surface=CacheSurface.bypass_cache
            )
            result_b = await solve_dependencies(
                request=Request(scope_b),
                dependant=dep,
                dependency_cache=cache,
                async_exit_stack=stack,
                embed_body_fields=False,
            )

            # Later sub-dependency in A still reads the frozen value (honor).
            result_a2 = await solve_dependencies(
                request=Request(scope_a),
                dependant=dep,
                dependency_cache=cache,
                async_exit_stack=stack,
                embed_body_fields=False,
            )

            key = next(iter(cache.keys()))
            return (
                result_a.values["value"],
                result_b.values["value"],
                result_a2.values["value"],
                cache[key],
                get_frozen_cache_keys(cache),
                calls["value"],
                rp_a.token,
            )

    _reset_calls()
    a1, b1, a2, cached, frozen_keys, total_calls, token_a = asyncio.run(scenario())
    assert a1 == 1
    assert b1 == 2  # B observably re-invoked the dependency
    assert a2 == 1  # owner request still reads the frozen value
    assert cached == 1  # the foreign bypass did not overwrite the cache
    assert total_calls == 2
    assert token_a is not None
    assert frozen_keys  # the key remains marked frozen


async def _freeze_endpoint(
    value: int = Depends(dep_value),
) -> None: ...  # pragma: no cover


def test_override_ignore_and_freeze_including_security() -> None:
    client = TestClient(override_app)
    override_app.dependency_overrides[get_data] = get_data_override
    override_app.dependency_overrides[get_user] = get_user_override
    try:
        # Baseline: overrides honored (Security + Depends alike).
        response = client.get("/whoami")
        assert response.json() == {
            "user": "override-user:['read']",
            "data": "override-data",
        }

        # ignore_overrides: original callables run for both.
        response = client.get("/whoami", headers={"x-override-surface": "ignore"})
        assert response.json() == {
            "user": "original-user:['read']",
            "data": "original-data",
        }

        # freeze_map: mapping mutated after push does not leak into the frame.
        response = client.get("/whoami", headers={"x-override-surface": "freeze"})
        assert response.json() == {
            "user": "override-user:['read']",
            "data": "override-data",
        }
    finally:
        override_app.dependency_overrides = {}


def test_bg_reject_middleware_and_late_push() -> None:
    client = TestClient(policy_app)

    bg_ran.clear()
    response = client.get("/bg-by-middleware", headers={"x-bg": "reject"})
    assert response.status_code == 200
    assert response.json()["rejected"] is True
    assert bg_ran == []  # rejected task never enqueued, never ran

    bg_ran.clear()
    response = client.get("/bg-reject-late-push")
    assert response.status_code == 200
    data = response.json()
    assert data["rejected"] is True
    assert data["queued"] == 1  # the early task survived the gate
    assert bg_ran == ["early"]  # it still ran after the response


def test_bg_suppress_run_keeps_queue_but_skips_execution() -> None:
    client = TestClient(policy_app)
    bg_ran.clear()
    response = client.get("/bg-by-middleware", headers={"x-bg": "suppress"})
    assert response.status_code == 200
    data = response.json()
    assert data["rejected"] is False
    # The task was admitted into the queue, not cleared...
    assert data["queued"] == 1
    assert data["suppressed"] == 1
    # ...and observably never ran after the response.
    assert bg_ran == []

    # Control: honor_queue runs the task after the response.
    bg_ran.clear()
    response = client.get("/bg-by-middleware", headers={"x-bg": "honor"})
    assert response.status_code == 200
    assert response.json()["rejected"] is False
    assert bg_ran == ["late"]


def test_gen_exit_hold_until_pop_vs_honor() -> None:
    client = TestClient(policy_app)

    gen_events.clear()
    response = client.get("/gen-honor")
    assert response.status_code == 200
    assert response.json()["closed_in_endpoint"] is False
    # honor: generator exits with the response, before middleware post-check.
    assert gen_events == ["enter", "exit", "after-response", "after-pop"]

    gen_events.clear()
    response = client.get("/gen-honor", headers={"x-gen": "hold"})
    assert response.status_code == 200
    # hold: response finished but the generator is still open; it exits at pop.
    assert gen_events == ["enter", "after-response", "exit", "after-pop"]


def test_gen_exit_force_close_now_and_late_push_void() -> None:
    client = TestClient(policy_app)

    gen_events.clear()
    response = client.get("/gen-force")
    assert response.status_code == 200
    assert response.json()["closed_at_pop"] is True
    # Closed exactly once, at pop, not again at response end.
    assert gen_events[:2] == ["enter", "exit"]
    assert gen_events.count("exit") == 1

    gen_events.clear()
    response = client.get("/gen-late-push")
    assert response.status_code == 200
    # The generator was already in the dependency body when force was pushed:
    # the frame cannot retroactively capture it, so it is not closed at pop.
    assert response.json()["closed_at_pop"] is False
    # It still closes normally with the response (before middleware post-check).
    assert gen_events == ["enter", "exit", "after-response", "after-pop"]


def test_hold_frame_unwinds_and_closes_generator_on_exception() -> None:
    client = TestClient(policy_app)
    gen_events.clear()
    with pytest.raises(RuntimeError):
        client.get("/gen-hold-error", headers={"x-gen": "hold"})
    # The exception did not skip the frame pop: the held generator was
    # closed on unwind instead of leaking.
    assert gen_events == ["enter", "exit"]


def test_effective_view_layering_and_lifo() -> None:
    client = TestClient(policy_app)
    response = client.get("/view")
    assert response.status_code == 200
    data = response.json()
    assert data["v0"] == [
        "honor_cache",
        "honor_overrides",
        "honor_queue",
        "honor_astack",
        0,
    ]
    # Dimensions are independent: outer cache bypass stays under inner frame.
    assert data["v1"] == ["reject_add", "bypass_cache", 1]
    assert data["v2"] == ["suppress_run", "bypass_cache", 2]
    # Inner pop restores the outer frame per dimension.
    assert data["v3"] == ["reject_add", "bypass_cache", 1]
    assert data["v4"] == ["honor_queue", 0]
    # Out-of-order pop is rejected.
    assert data["bad_pop"] is True


def test_empty_stack_matches_current_behavior() -> None:
    client = TestClient(policy_app)
    _reset_calls()
    response = client.get("/plain")
    assert response.status_code == 200
    assert response.json() == {"value": 1, "default_bg": "honor_queue"}
    view = get_policy_view()
    assert view.depth == 0
    assert view.held_generator_count == 0


def test_push_without_connection_uses_context_bound_stack() -> None:
    client = TestClient(policy_app)
    bg_ran.clear()
    response = client.get("/bg-contextvar")
    assert response.status_code == 200
    assert response.json() == {"rejected": True}
    assert bg_ran == []


def test_websocket_override_fork_and_bg_reject() -> None:
    client = TestClient(ws_app)
    ws_app.dependency_overrides[ws_dep] = ws_dep_override
    try:
        with client.websocket_connect("/ws") as websocket:
            assert websocket.receive_text() == "ws-override"
        with client.websocket_connect(
            "/ws", headers={"x-ws-override": "ignore"}
        ) as websocket:
            assert websocket.receive_text() == "ws-original"
    finally:
        ws_app.dependency_overrides = {}

    bg_ran.clear()
    with client.websocket_connect("/ws-bg") as websocket:
        assert websocket.receive_json() == {"rejected": True, "queued": 1}
    # WebSocket has no HTTP-style post-response run; the early task is never
    # auto-executed (and suppression is not faked as such a run).
    assert bg_ran == []


def test_concurrent_requests_do_not_cross_stacks() -> None:
    async def scenario() -> tuple[Any, Any]:
        transport = httpx.ASGITransport(app=concurrent_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            result_a, result_b = await asyncio.gather(
                client.get(
                    "/concurrent",
                    headers={"x-tag": "a", "x-bg": "reject"},
                ),
                client.get(
                    "/concurrent",
                    headers={"x-tag": "b", "x-bg": "honor"},
                ),
            )
        return result_a.json(), result_b.json()

    bg_ran.clear()
    barrier_events.clear()
    data_a, data_b = asyncio.run(scenario())
    assert data_a["admitted"] is False
    assert data_b["admitted"] is True
    # Distinct request-bound stacks.
    assert data_a["token"] != data_b["token"]
    # Only the honor request's task ran after its response; the reject
    # request's gate did not leak across the concurrent connection.
    assert bg_ran == ["b"]
