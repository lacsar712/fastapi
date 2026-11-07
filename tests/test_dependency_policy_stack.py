import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any

import pytest
from fastapi import (
    AstackRelease,
    BackgroundTasks,
    Depends,
    FastAPI,
    OverrideLatch,
    Request,
    Security,
    SolveReuse,
    TaskAdmit,
    TaskRefusedError,
    WebSocket,
    dependency_policy,
    get_effective_dependency_policy,
    pop_dependency_policy,
    push_dependency_policy,
)
from fastapi.dependencies.utils import get_dependant, solve_dependencies
from fastapi.dependency_policy import (
    POLICY_STACK_SCOPE_KEY,
    DependencyPolicyError,
)
from fastapi.security.oauth2 import SecurityScopes
from fastapi.testclient import TestClient
from starlette.middleware import Middleware
from starlette.requests import HTTPConnection
from starlette.types import Scope


def _view_dict(view: Any) -> dict[str, Any]:
    return {
        "depth": view.depth,
        "solve_reuse": view.solve_reuse.value,
        "override_latch": view.override_latch.value,
        "task_admit": view.task_admit.value,
        "astack_release": view.astack_release.value,
        "admitted_tasks": view.admitted_tasks,
        "queued_tasks": view.queued_tasks,
        "muted_tasks": view.muted_tasks,
        "registered_generators": view.registered_generators,
    }


# ---------------------------------------------------------------------------
# Shared per-request counters and diamond dependency (use_cache=True)
# ---------------------------------------------------------------------------

_counts_lock = threading.Lock()
_counts: dict[str, int] = {}

_recorder_lock = threading.Lock()
_recorder: list[str] = []


def _reset_state() -> None:
    with _counts_lock:
        _counts.clear()
    with _recorder_lock:
        _recorder.clear()


def record_task(marker: str) -> None:
    with _recorder_lock:
        _recorder.append(marker)


def common_dep(conn: HTTPConnection) -> int:
    rid = conn.headers.get("x-rid", conn.query_params.get("rid", "default"))
    with _counts_lock:
        n = _counts.get(rid, 0) + 1
        _counts[rid] = n
    return n


def branch_a_dep(value: Annotated[int, Depends(common_dep)]) -> int:
    return value


def branch_b_dep(value: Annotated[int, Depends(common_dep)]) -> int:
    return value


def push_force_dep(conn: HTTPConnection) -> Any:
    # Push while recursion has already begun: remaining solving must recompute.
    if conn.headers.get("x-force") == "1":
        return push_dependency_policy(conn.scope, solve_reuse=SolveReuse.FORCE_REINVOKE)
    return None


def push_skip_dep(conn: HTTPConnection) -> Any:
    if conn.headers.get("x-skip") == "1":
        return push_dependency_policy(conn.scope, override_latch=OverrideLatch.SKIP_MAP)
    return None


def plain_dep() -> str:
    return "original"


def plain_override() -> str:
    return "override"


def secure_dep(security_scopes: SecurityScopes) -> tuple[str, tuple[str, ...]]:
    return ("original", tuple(security_scopes.scopes))


def secure_override() -> str:
    return "override-security"


app = FastAPI()


@app.get("/plain")
async def plain_route(request: Request) -> dict[str, Any]:
    view = get_effective_dependency_policy(request)
    return {
        "view": _view_dict(view),
        "stack_attached": POLICY_STACK_SCOPE_KEY in request.scope,
    }


@app.get("/diamond")
async def diamond_route(
    conn: HTTPConnection,
    pushed: Annotated[Any, Depends(push_force_dep)],
    a: Annotated[int, Depends(branch_a_dep)],
    b: Annotated[int, Depends(branch_b_dep)],
) -> dict[str, Any]:
    rid = conn.headers.get("x-rid", "default")
    depth = get_effective_dependency_policy(conn).depth
    if pushed is not None:
        await pop_dependency_policy(conn, pushed)
    return {"a": a, "b": b, "count": _counts[rid], "depth": depth}


# --- pin_cached through nested manual solve_dependencies calls -------------

pin_calls = {"n": 0}


def pin_dep() -> int:
    pin_calls["n"] += 1
    return pin_calls["n"]


def pin_wrapper(c: Annotated[int, Depends(pin_dep)]) -> int:  # pragma: no cover
    return c


@app.get("/pin")
async def pin_route(request: Request) -> dict[str, Any]:
    cache: dict[Any, Any] = {}
    astack = request.scope["fastapi_inner_astack"]

    async def solve_once() -> int:
        dependant = get_dependant(path="/pin", call=pin_wrapper)
        solved = await solve_dependencies(
            request=request,
            dependant=dependant,
            dependency_overrides_provider=request.app,
            dependency_cache=cache,
            async_exit_stack=astack,
            embed_body_fields=False,
        )
        return int(solved.values["c"])

    pin_frame = push_dependency_policy(request, solve_reuse=SolveReuse.PIN_CACHED)
    v1 = await solve_once()
    force_frame = push_dependency_policy(request, solve_reuse=SolveReuse.FORCE_REINVOKE)
    v2 = await solve_once()
    await pop_dependency_policy(request, force_frame)
    v3 = await solve_once()
    await pop_dependency_policy(request, pin_frame)
    v4 = await solve_once()
    return {"v1": v1, "v2": v2, "v3": v3, "v4": v4, "calls": pin_calls["n"]}


# --- BackgroundTasks: refuse / mute -----------------------------------------


def queueing_dep(bt: BackgroundTasks) -> str:
    bt.add_task(record_task, "already-queued")
    return "queued"


@app.get("/refuse")
async def refuse_route(request: Request, bt: BackgroundTasks) -> dict[str, Any]:
    frame = push_dependency_policy(request, task_admit=TaskAdmit.REFUSE_ENQUEUE)
    try:
        bt.add_task(record_task, "should-not-be-admitted")
        result = "enqueued"
    except TaskRefusedError:
        result = "refused"
    await pop_dependency_policy(request, frame)
    return {"result": result, "queued": len(bt.tasks)}


@app.get("/late-refuse")
async def late_refuse_route(
    request: Request,
    _queued: Annotated[str, Depends(queueing_dep)],
) -> dict[str, Any]:
    try:
        push_dependency_policy(request, task_admit=TaskAdmit.REFUSE_ENQUEUE)
        rejected = False
    except DependencyPolicyError:
        rejected = True
    return {"rejected": rejected}


@app.get("/mute")
async def mute_route(request: Request, bt: BackgroundTasks) -> dict[str, Any]:
    frame = push_dependency_policy(request, task_admit=TaskAdmit.MUTE_AFTER_RESPONSE)
    bt.add_task(record_task, "muted-task")
    await pop_dependency_policy(request, frame)
    bt.add_task(record_task, "normal-task")
    return {"queued": len(bt.tasks)}


@app.get("/mute-late")
async def mute_late_route(request: Request, bt: BackgroundTasks) -> dict[str, Any]:
    bt.add_task(record_task, "early-normal-task")
    frame = push_dependency_policy(request, task_admit=TaskAdmit.MUTE_AFTER_RESPONSE)
    bt.add_task(record_task, "later-muted-task")
    await pop_dependency_policy(request, frame)
    return {"queued": len(bt.tasks)}


# --- override_latch: skip_map -----------------------------------------------


@app.get("/skip")
async def skip_route(
    conn: HTTPConnection,
    pushed: Annotated[Any, Depends(push_skip_dep)],
    plain: Annotated[str, Depends(plain_dep)],
    secure: Annotated[
        tuple[str, tuple[str, ...]], Security(secure_dep, scopes=["read"])
    ],
) -> dict[str, Any]:
    if pushed is not None:
        await pop_dependency_policy(conn, pushed)
    return {"plain": plain, "secure": secure}


# --- effective view ---------------------------------------------------------


@app.get("/view")
async def view_route(request: Request) -> dict[str, Any]:
    before = _view_dict(get_effective_dependency_policy(request))
    frame = push_dependency_policy(
        request,
        solve_reuse=SolveReuse.PIN_CACHED,
        override_latch=OverrideLatch.SKIP_MAP,
        task_admit=TaskAdmit.REFUSE_ENQUEUE,
        astack_release=AstackRelease.DEFER_UNTIL_POP,
    )
    inner = _view_dict(get_effective_dependency_policy(request))
    await pop_dependency_policy(request, frame)
    after = _view_dict(get_effective_dependency_policy(request))
    return {"before": before, "inner": inner, "after": after}


@app.get("/nested")
async def nested_route(request: Request) -> dict[str, Any]:
    outer = push_dependency_policy(request, solve_reuse=SolveReuse.FORCE_REINVOKE)
    outer_view = get_effective_dependency_policy(request)
    inner = push_dependency_policy(request, astack_release=AstackRelease.CLOSE_ON_POP)
    inner_view = get_effective_dependency_policy(request)
    await pop_dependency_policy(request, inner)
    back_to_outer = get_effective_dependency_policy(request)
    await pop_dependency_policy(request, outer)
    back_to_empty = get_effective_dependency_policy(request)
    return {
        "outer": outer_view.solve_reuse.value,
        "outer_depth": outer_view.depth,
        "inner_depth": inner_view.depth,
        "inner_release": inner_view.astack_release.value,
        "back_outer_reuse": back_to_outer.solve_reuse.value,
        "back_outer_release": back_to_outer.astack_release.value,
        "back_depth": back_to_outer.depth,
        "empty_depth": back_to_empty.depth,
    }


@app.get("/view-tasks")
async def view_tasks_route(request: Request, bt: BackgroundTasks) -> dict[str, Any]:
    frame = push_dependency_policy(request, task_admit=TaskAdmit.MUTE_AFTER_RESPONSE)
    bt.add_task(record_task, "view-muted")
    during = _view_dict(get_effective_dependency_policy(request))
    await pop_dependency_policy(request, frame)
    return {"during": during}


# --- astack_release: close_on_pop pushed before a generator sibling --------

gen_state: dict[str, Any] = {"alive": False, "events": []}


async def request_gen(request: Request) -> Any:
    gen_state["events"].append("enter")
    gen_state["alive"] = True
    try:
        yield "g"
    finally:
        gen_state["alive"] = False
        gen_state["events"].append("exit")


def push_close_dep(request: Request) -> Any:
    return push_dependency_policy(
        request.scope, astack_release=AstackRelease.CLOSE_ON_POP
    )


@app.get("/close-on-pop")
async def close_on_pop_route(
    request: Request,
    pushed: Annotated[Any, Depends(push_close_dep)],
    _g: Annotated[str, Depends(request_gen)],
) -> dict[str, Any]:
    before = {"alive": gen_state["alive"], "events": list(gen_state["events"])}
    await pop_dependency_policy(request, pushed)
    after = {"alive": gen_state["alive"], "events": list(gen_state["events"])}
    return {"before": before, "after": after}


@app.get("/close-late")
async def close_late_route(
    request: Request,
    _g: Annotated[str, Depends(request_gen)],
) -> dict[str, Any]:
    try:
        push_dependency_policy(request, astack_release=AstackRelease.CLOSE_ON_POP)
        rejected = False
    except DependencyPolicyError:
        rejected = True
    return {
        "rejected": rejected,
        "alive": gen_state["alive"],
        "events": list(gen_state["events"]),
    }


client = TestClient(app)
app.dependency_overrides.update(
    {plain_dep: plain_override, secure_dep: secure_override}
)


# --- freeze_at_enter through raw ASGI middleware ----------------------------


def freeze_target_dep() -> str:
    return "original-frozen"


def freeze_override_dep() -> str:
    return "override-frozen"


def freeze_mutator_add(request: Request) -> None:
    request.app.dependency_overrides[freeze_target_dep] = freeze_override_dep


def freeze_mutator_del(request: Request) -> None:
    request.app.dependency_overrides.pop(freeze_target_dep, None)


class FreezeMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        saved = dict(scope["app"].dependency_overrides)
        try:
            async with dependency_policy(
                scope, override_latch=OverrideLatch.FREEZE_AT_ENTER
            ):
                await self.app(scope, receive, send)
        finally:
            scope["app"].dependency_overrides.clear()
            scope["app"].dependency_overrides.update(saved)


freeze_add_app = FastAPI(middleware=[Middleware(FreezeMiddleware)])


@freeze_add_app.get("/freeze")
async def freeze_add_route(
    _mutation: Annotated[None, Depends(freeze_mutator_add)],
    target: Annotated[str, Depends(freeze_target_dep)],
) -> dict[str, str]:
    return {"target": target}


freeze_del_app = FastAPI(middleware=[Middleware(FreezeMiddleware)])
freeze_del_app.dependency_overrides[freeze_target_dep] = freeze_override_dep


@freeze_del_app.get("/freeze")
async def freeze_del_route(
    _mutation: Annotated[None, Depends(freeze_mutator_del)],
    target: Annotated[str, Depends(freeze_target_dep)],
) -> dict[str, str]:
    return {"target": target}


# --- astack_release: defer_until_pop vs honor through middleware ------------


release_state: dict[str, Any] = {"alive": False, "events": []}


async def defer_request_gen() -> Any:
    release_state["events"].append("enter")
    release_state["alive"] = True
    try:
        yield "g"
    finally:
        release_state["alive"] = False
        release_state["events"].append("exit")


def defer_sync_gen() -> Any:
    release_state["events"].append("sync-enter")
    try:
        yield "s"
    finally:
        release_state["events"].append("sync-exit")


def defer_background() -> None:
    release_state["events"].append(("bg", release_state["alive"]))


class ReleaseMiddleware:
    def __init__(self, app: Any, release: AstackRelease, result: dict) -> None:
        self.app = app
        self.release = release
        self.result = result

    async def __call__(self, scope: Scope, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        async with dependency_policy(scope, astack_release=self.release):
            await self.app(scope, receive, send)
            self.result["after_response"] = {
                "alive": release_state["alive"],
                "events": list(release_state["events"]),
            }
        self.result["after_pop"] = {
            "alive": release_state["alive"],
            "events": list(release_state["events"]),
        }


def _build_release_app(release: AstackRelease, result: dict) -> FastAPI:
    release_app = FastAPI(
        middleware=[Middleware(ReleaseMiddleware, release=release, result=result)]
    )

    @release_app.get("/g")
    async def g_route(
        bt: BackgroundTasks,
        g: Annotated[str, Depends(defer_request_gen)],
        s: Annotated[str, Depends(defer_sync_gen, scope="function")],
    ) -> dict[str, Any]:
        bt.add_task(defer_background)
        return {
            "alive": release_state["alive"],
            "events": list(release_state["events"]),
        }

    return release_app


# --- exception path closes frame-owned generators --------------------------

error_state: dict[str, Any] = {"alive": False, "events": []}


async def error_gen() -> Any:
    error_state["events"].append("enter")
    error_state["alive"] = True
    try:
        yield "g"
    finally:
        error_state["alive"] = False
        error_state["events"].append("exit")


class CloseOnErrorMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        async with dependency_policy(scope, astack_release=AstackRelease.CLOSE_ON_POP):
            await self.app(scope, receive, send)


error_app = FastAPI(middleware=[Middleware(CloseOnErrorMiddleware)])


@error_app.get("/boom")
async def boom_route(_g: Annotated[str, Depends(error_gen)]) -> None:
    raise RuntimeError("boom")


# --- WebSocket ---------------------------------------------------------------

ws_app = FastAPI()


def ws_push_force(conn: HTTPConnection) -> Any:
    if conn.query_params.get("force") == "1":
        return push_dependency_policy(conn.scope, solve_reuse=SolveReuse.FORCE_REINVOKE)
    return None


@ws_app.websocket("/ws")
async def ws_diamond(
    websocket: WebSocket,
    pushed: Annotated[Any, Depends(ws_push_force)],
    a: Annotated[int, Depends(branch_a_dep)],
    b: Annotated[int, Depends(branch_b_dep)],
) -> None:
    if pushed is not None:
        await pop_dependency_policy(websocket, pushed)
    rid = websocket.query_params.get("rid", "ws")
    await websocket.accept()
    await websocket.send_json({"a": a, "b": b, "count": _counts[rid]})
    await websocket.close()


@ws_app.websocket("/ws-refuse")
async def ws_refuse(websocket: WebSocket, bt: BackgroundTasks) -> None:
    await websocket.accept()
    frame = push_dependency_policy(websocket, task_admit=TaskAdmit.REFUSE_ENQUEUE)
    try:
        bt.add_task(record_task, "ws-should-not-run")
        result = "enqueued"
    except TaskRefusedError:
        result = "refused"
    await pop_dependency_policy(websocket, frame)
    await websocket.send_json({"result": result, "queued": len(bt.tasks)})
    await websocket.close()


# --- concurrency isolation app ----------------------------------------------

gate_lock = threading.Lock()
gate_events: list[Any] = []


async def concurrency_gate() -> None:
    event = asyncio.Event()
    with gate_lock:
        gate_events.append(event)
        number = len(gate_events)
    if number == 1:
        await asyncio.wait_for(event.wait(), timeout=10)
    else:
        gate_events[0].set()
        await asyncio.sleep(0)


concurrent_app = FastAPI()


@concurrent_app.get("/solve")
async def concurrent_solve_route(
    conn: HTTPConnection,
    pushed: Annotated[Any, Depends(push_force_dep)],
    a: Annotated[int, Depends(branch_a_dep)],
    b: Annotated[int, Depends(branch_b_dep)],
) -> dict[str, Any]:
    rid = conn.headers["x-rid"]
    await concurrency_gate()
    depth = get_effective_dependency_policy(conn).depth
    if pushed is not None:
        await pop_dependency_policy(conn, pushed)
    return {"a": a, "b": b, "count": _counts[rid], "depth": depth}


@concurrent_app.get("/task")
async def concurrent_task_route(
    conn: HTTPConnection, bt: BackgroundTasks
) -> dict[str, Any]:
    rid = conn.headers["x-rid"]
    frame = None
    if conn.headers.get("x-admit") == "refuse":
        frame = push_dependency_policy(conn.scope, task_admit=TaskAdmit.REFUSE_ENQUEUE)
    await concurrency_gate()
    try:
        bt.add_task(record_task, rid)
        result = "enqueued"
    except TaskRefusedError:
        result = "refused"
    if frame is not None:
        await pop_dependency_policy(conn, frame)
    return {"rid": rid, "result": result}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_stack_keeps_legacy_behavior() -> None:
    _reset_state()
    response = client.get("/plain")
    assert response.status_code == 200
    data = response.json()
    assert data["stack_attached"] is False
    assert data["view"] == _view_dict(get_effective_dependency_policy({}))
    assert data["view"]["depth"] == 0
    assert data["view"]["solve_reuse"] == "honor_reuse"
    assert data["view"]["task_admit"] == "honor_enqueue"
    assert data["view"]["astack_release"] == "honor_response_exit"


def test_http_honor_reuse_invokes_once() -> None:
    _reset_state()
    response = client.get("/diamond", headers={"x-rid": "honor"})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["a"] == data["b"] == 1
    assert data["depth"] == 0


def test_http_force_reinvoke_after_recursion_started() -> None:
    _reset_state()
    response = client.get("/diamond", headers={"x-rid": "force", "x-force": "1"})
    assert response.status_code == 200
    data = response.json()
    # The frame was pushed from a dependency while solving was already running,
    # yet the remaining sub-dependencies re-invoked instead of hitting cache.
    assert data["count"] == 2
    assert data["a"] == 1
    assert data["b"] == 2
    assert data["depth"] == 1


def test_pin_cached_blocks_force_overwrite_within_request() -> None:
    pin_calls["n"] = 0
    response = client.get("/pin")
    assert response.status_code == 200
    data = response.json()
    assert data == {"v1": 1, "v2": 2, "v3": 1, "v4": 1, "calls": 2}


def test_skip_map_runs_original_for_depends_and_security() -> None:
    response = client.get("/skip", headers={"x-skip": "1"})
    assert response.status_code == 200
    data = response.json()
    assert data["plain"] == "original"
    assert data["secure"] == ["original", ["read"]]

    response = client.get("/skip")
    assert response.status_code == 200
    data = response.json()
    assert data["plain"] == "override"
    assert data["secure"] == "override-security"


def test_freeze_at_enter_ignores_mapping_added_after_push() -> None:
    response = TestClient(freeze_add_app).get("/freeze")
    assert response.status_code == 200
    assert response.json() == {"target": "original-frozen"}
    # The middleware restored the mapping; no leftover override.
    assert freeze_target_dep not in freeze_add_app.dependency_overrides


def test_freeze_at_enter_keeps_mapping_deleted_after_push() -> None:
    response = TestClient(freeze_del_app).get("/freeze")
    assert response.status_code == 200
    assert response.json() == {"target": "override-frozen"}
    assert freeze_del_app.dependency_overrides[freeze_target_dep] is (
        freeze_override_dep
    )


def test_refuse_enqueue_fails_and_keeps_queue_intact() -> None:
    _reset_state()
    response = client.get("/refuse")
    assert response.status_code == 200
    data = response.json()
    assert data == {"result": "refused", "queued": 0}
    assert _recorder == []


def test_refuse_enqueue_push_after_queue_is_rejected() -> None:
    _reset_state()
    response = client.get("/late-refuse")
    assert response.status_code == 200
    assert response.json() == {"rejected": True}
    # The already queued task was not cleared by the refused gate and ran.
    assert _recorder == ["already-queued"]


def test_mute_after_response_skips_execution_but_keeps_queue() -> None:
    _reset_state()
    response = client.get("/mute")
    assert response.status_code == 200
    assert response.json() == {"queued": 2}
    # Only the task enqueuted outside the muted frame ran; muted task remains
    # observable in the queue instead of being cleared.
    assert _recorder == ["normal-task"]


def test_mute_after_response_late_push_only_mutes_later_tasks() -> None:
    _reset_state()
    response = client.get("/mute-late")
    assert response.status_code == 200
    assert response.json() == {"queued": 2}
    assert _recorder == ["early-normal-task"]


def test_close_on_pop_closes_generator_before_response() -> None:
    gen_state["alive"] = False
    gen_state["events"] = []
    response = client.get("/close-on-pop")
    assert response.status_code == 200
    data = response.json()
    assert data["before"] == {"alive": True, "events": ["enter"]}
    assert data["after"] == {
        "alive": False,
        "events": ["enter", "exit"],
    }


def test_close_on_pop_after_generator_entered_is_rejected() -> None:
    gen_state["alive"] = False
    gen_state["events"] = []
    response = client.get("/close-late")
    assert response.status_code == 200
    data = response.json()
    assert data["rejected"] is True
    assert data["alive"] is True
    assert data["events"] == ["enter"]


def test_defer_until_pop_vs_honor_response_exit() -> None:
    defer_result: dict[str, Any] = {}
    release_state["alive"] = False
    release_state["events"] = []
    defer_app = _build_release_app(AstackRelease.DEFER_UNTIL_POP, defer_result)
    response = TestClient(defer_app).get("/g")
    assert response.status_code == 200
    assert response.json()["events"] == ["enter", "sync-enter"]
    # Response (and background tasks) done, generators still open.
    assert defer_result["after_response"] == {
        "alive": True,
        "events": ["enter", "sync-enter", ("bg", True)],
    }
    # Pop closed the frame-owned generator stacks.
    assert defer_result["after_pop"]["alive"] is False
    assert defer_result["after_pop"]["events"] == [
        "enter",
        "sync-enter",
        ("bg", True),
        "sync-exit",
        "exit",
    ]

    honor_result: dict[str, Any] = {}
    release_state["alive"] = False
    release_state["events"] = []
    honor_app = _build_release_app(AstackRelease.HONOR_RESPONSE_EXIT, honor_result)
    response = TestClient(honor_app).get("/g")
    assert response.status_code == 200
    # Function scope closed before the response, request scope right after,
    # all before the middleware regained control.
    assert honor_result["after_response"]["alive"] is False
    assert honor_result["after_response"]["events"] == [
        "enter",
        "sync-enter",
        "sync-exit",
        ("bg", True),
        "exit",
    ]
    assert honor_result["after_pop"] == honor_result["after_response"]


def test_exception_path_pops_frame_and_closes_generators() -> None:
    error_state["alive"] = False
    error_state["events"] = []
    with TestClient(error_app, raise_server_exceptions=False) as error_client:
        response = error_client.get("/boom")
    assert response.status_code == 500
    assert error_state["alive"] is False
    assert error_state["events"] == ["enter", "exit"]


def test_websocket_solve_reuse_fork() -> None:
    _reset_state()
    with TestClient(ws_app) as ws_client:
        with ws_client.websocket_connect("/ws?rid=wshonor") as ws:
            assert ws.receive_json() == {"a": 1, "b": 1, "count": 1}
        with ws_client.websocket_connect("/ws?rid=wsforce&force=1") as ws:
            assert ws.receive_json() == {"a": 1, "b": 2, "count": 2}


def test_websocket_refuse_enqueue() -> None:
    _reset_state()
    with TestClient(ws_app) as ws_client:
        with ws_client.websocket_connect("/ws-refuse") as ws:
            assert ws.receive_json() == {"result": "refused", "queued": 0}
    assert _recorder == []


def test_effective_view_reflects_and_restores_frames() -> None:
    response = client.get("/view")
    assert response.status_code == 200
    data = response.json()
    assert data["before"]["depth"] == 0
    assert data["after"]["depth"] == 0
    assert data["inner"] == {
        "depth": 1,
        "solve_reuse": "pin_cached",
        "override_latch": "skip_map",
        "task_admit": "refuse_enqueue",
        "astack_release": "defer_until_pop",
        "admitted_tasks": 0,
        "queued_tasks": 0,
        "muted_tasks": 0,
        "registered_generators": 0,
    }


def test_nested_frames_inner_pop_returns_to_outer() -> None:
    response = client.get("/nested")
    assert response.status_code == 200
    assert response.json() == {
        "outer": "force_reinvoke",
        "outer_depth": 1,
        "inner_depth": 2,
        "inner_release": "close_on_pop",
        "back_outer_reuse": "force_reinvoke",
        "back_outer_release": "honor_response_exit",
        "back_depth": 1,
        "empty_depth": 0,
    }


def test_effective_view_task_counts() -> None:
    _reset_state()
    response = client.get("/view-tasks")
    assert response.status_code == 200
    assert response.json()["during"] == {
        "depth": 1,
        "solve_reuse": "honor_reuse",
        "override_latch": "honor_map",
        "task_admit": "mute_after_response",
        "astack_release": "honor_response_exit",
        "admitted_tasks": 1,
        "queued_tasks": 1,
        "muted_tasks": 1,
        "registered_generators": 0,
    }
    assert _recorder == []


def test_pop_empty_stack_raises_and_scopes_are_independent() -> None:
    first_scope: Scope = {}
    second_scope: Scope = {}
    first_frame = push_dependency_policy(
        first_scope, solve_reuse=SolveReuse.FORCE_REINVOKE
    )
    # A frame on one scope cannot be popped through another scope's stack.
    with pytest.raises(DependencyPolicyError):
        asyncio.run(pop_dependency_policy(second_scope, first_frame))
    asyncio.run(pop_dependency_policy(first_scope, first_frame))
    with pytest.raises(DependencyPolicyError):
        asyncio.run(pop_dependency_policy(first_scope))


def test_concurrent_requests_solve_reuse_stacks_isolated() -> None:
    _reset_state()
    gate_events.clear()
    with ThreadPoolExecutor(max_workers=2) as pool:
        with TestClient(concurrent_app) as concurrent_client:
            force_future = pool.submit(
                concurrent_client.get,
                "/solve",
                headers={"x-rid": "cforce", "x-force": "1"},
            )
            honor_future = pool.submit(
                concurrent_client.get,
                "/solve",
                headers={"x-rid": "chonor"},
            )
            force_response = force_future.result(timeout=30)
            honor_response = honor_future.result(timeout=30)
    assert force_response.status_code == 200
    assert honor_response.status_code == 200
    force_data = force_response.json()
    honor_data = honor_response.json()
    assert force_data == {"a": 1, "b": 2, "count": 2, "depth": 1}
    assert honor_data == {"a": 1, "b": 1, "count": 1, "depth": 0}
    assert set(_counts) == {"cforce", "chonor"}


def test_concurrent_requests_task_admit_stacks_isolated() -> None:
    _reset_state()
    gate_events.clear()
    with ThreadPoolExecutor(max_workers=2) as pool:
        with TestClient(concurrent_app) as concurrent_client:
            refuse_future = pool.submit(
                concurrent_client.get,
                "/task",
                headers={"x-rid": "trefuse", "x-admit": "refuse"},
            )
            honor_future = pool.submit(
                concurrent_client.get,
                "/task",
                headers={"x-rid": "thonor", "x-admit": "honor"},
            )
            refuse_response = refuse_future.result(timeout=30)
            honor_response = honor_future.result(timeout=30)
    assert refuse_response.json() == {"rid": "trefuse", "result": "refused"}
    assert honor_response.json() == {"rid": "thonor", "result": "enqueued"}
    assert _recorder == ["thonor"]
