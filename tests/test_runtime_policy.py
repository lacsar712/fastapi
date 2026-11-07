"""Tests for the request-scoped runtime policy stack.

Covers the three independent axes (``dep_admit``, ``serialize_gate`` and
``background_gate``), in-flight semantics, force cancellation of background
tasks, per-request isolation on the same event loop, exception-safe pops and
decoupling from the permanent app/router defaults, OpenAPI and static mounts.
"""

import asyncio
import warnings

import pytest
from fastapi import (
    BackgroundPolicy,
    BackgroundTasks,
    DepAdmitPolicy,
    Depends,
    FastAPI,
    PolicyView,
    RequestPolicyError,
    Security,
    SerializationDenied,
    SerializePolicy,
)
from fastapi.datastructures import DefaultPlaceholder
from fastapi.exceptions import FastAPIDeprecationWarning
from fastapi.responses import PlainTextResponse
from fastapi.runtime_policy import (
    BackgroundEnqueueDenied,
    CancelledBackgroundTask,
    DependencyResolutionDenied,
    RequestPolicyFrame,
    admit_dependency,
    admit_serialization,
    get_policy_view,
    policy_frame,
    pop_policy,
    push_policy,
)
from fastapi.security import APIKeyHeader
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from starlette.responses import JSONResponse as StarletteJSONResponse

pytestmark = pytest.mark.filterwarnings(
    "ignore:ORJSONResponse is deprecated:fastapi.exceptions.FastAPIDeprecationWarning",
    "ignore:UJSONResponse is deprecated:fastapi.exceptions.FastAPIDeprecationWarning",
)


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.exception_handler(RequestPolicyError)
    async def policy_error_handler(request, exc: RequestPolicyError):  # type: ignore[no-untyped-def]
        kind = getattr(exc, "kind", "")
        detail = f"denied:{exc.axis}:{exc.policy}:{kind}"
        # Plain text on purpose: rendering this error surface must not itself
        # cross the JSON serialization gate.
        return PlainTextResponse(detail, status_code=503)

    return app


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Empty stack: pinned behavior
# ---------------------------------------------------------------------------


def test_empty_stack_behaves_like_pin():
    app = _make_app()
    ran = []

    def task():
        ran.append("task")

    async def dep():
        return "dep-value"

    @app.get("/")
    async def root(value: str = Depends(dep)):
        return {"value": value}

    @app.get("/bg")
    async def with_bg(tasks: BackgroundTasks):
        tasks.add_task(task)
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/").json() == {"value": "dep-value"}
    assert client.get("/bg").status_code == 200
    assert ran == ["task"]
    view = get_policy_view()
    assert view == PolicyView(
        dep_admit=DepAdmitPolicy.HONOR_CFG,
        serialize_gate=SerializePolicy.HONOR_CFG,
        background_gate=BackgroundPolicy.HONOR_CFG,
        dep_admit_remaining=None,
        depth=0,
    )


def test_openapi_and_defaults_untouched():
    app = _make_app()

    @app.get("/items/")
    async def items():
        return []

    default_class = app.router.default_response_class
    assert isinstance(default_class, DefaultPlaceholder)
    assert default_class.value is StarletteJSONResponse

    client = TestClient(app)
    schema = client.get("/openapi.json").json()
    assert "/items/" in schema["paths"]
    assert app.dependency_overrides == {}


# ---------------------------------------------------------------------------
# dep_admit
# ---------------------------------------------------------------------------


def test_deny_new_rejects_new_dependencies_but_keeps_in_flight_one():
    app = _make_app()
    effects = []

    async def first():
        # This dependency was already admitted before the frame exists.  The
        # frame is intentionally left active for the remaining siblings.
        push_policy(dep_admit=DepAdmitPolicy.DENY_NEW)
        effects.append("first-started")
        await asyncio.sleep(0)
        effects.append("first-finished")
        # Frame intentionally left active for the remaining siblings.
        return "first"

    async def second():  # pragma: no cover - must never be admitted
        effects.append("second")
        return "second"

    @app.get("/")
    async def root(
        a: str = Depends(first),
        b: str = Depends(second),
    ):
        return {"a": a, "b": b}

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/")
    assert response.status_code == 503
    assert response.text.startswith("denied:dep_admit:deny_new")
    # The in-flight dependency ran to completion with its real value logic;
    # it was neither cancelled nor rewritten into an empty success.
    assert effects == ["first-started", "first-finished"]


def test_deny_new_covers_security_and_sub_dependencies():
    app = _make_app()
    scheme = APIKeyHeader(name="X-Key", auto_error=False)

    async def gatekeeper(token: str | None = Security(scheme)):
        return token

    async def blocker():
        push_policy(dep_admit=DepAdmitPolicy.DENY_NEW)
        return True

    @app.get("/")
    async def root(
        blocked: bool = Depends(blocker),
        token: str | None = Depends(gatekeeper),
    ):
        return {"token": token}

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/", headers={"X-Key": "secret"})
    assert response.status_code == 503
    assert "dep_admit" in response.text


def test_one_more_allows_exactly_one_additional_admission():
    app = _make_app()
    effects = []

    async def first():
        push_policy(dep_admit=DepAdmitPolicy.ONE_MORE)
        effects.append("first")
        return 1

    async def second():
        view = get_policy_view()
        assert view.dep_admit is DepAdmitPolicy.ONE_MORE
        assert view.dep_admit_remaining == 0  # consumed by this admission
        effects.append("second")
        return 2

    async def third():  # pragma: no cover - third admission is refused
        effects.append("third")
        return 3

    @app.get("/")
    async def root(
        a: int = Depends(first),
        b: int = Depends(second),
        c: int = Depends(third),
    ):
        return {"sum": a + b + c}

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/")
    assert response.status_code == 503
    assert "dep_admit:one_more" in response.text
    assert effects == ["first", "second"]


def test_one_more_every_admission_rereads_current_top():
    # Outer one_more is consumed; an inner one_more pushed afterwards gives a
    # fresh budget because admissions re-read the current top.
    outer = push_policy(dep_admit=DepAdmitPolicy.ONE_MORE)
    assert get_policy_view().dep_admit_remaining == 1
    admit_dependency()  # the single allowed admission
    assert get_policy_view().dep_admit_remaining == 0
    with pytest.raises(DependencyResolutionDenied):
        admit_dependency()

    inner = push_policy(dep_admit=DepAdmitPolicy.ONE_MORE)
    assert get_policy_view().dep_admit_remaining == 1
    admit_dependency()  # allowed by the new top
    with pytest.raises(DependencyResolutionDenied):
        admit_dependency()
    pop_policy(inner)
    # Inner pop restores the exhausted outer frame.
    assert get_policy_view().depth == 1
    with pytest.raises(DependencyResolutionDenied):
        admit_dependency()
    pop_policy(outer)
    assert get_policy_view().depth == 0


def test_honor_frame_is_transparent_to_outer_deny():
    outer = push_policy(dep_admit=DepAdmitPolicy.DENY_NEW)
    inner = push_policy()  # all axes honor_cfg
    with pytest.raises(DependencyResolutionDenied):
        admit_dependency()
    pop_policy(inner)
    with pytest.raises(DependencyResolutionDenied):
        admit_dependency()
    pop_policy(outer)


def test_dependency_overrides_still_apply_under_honor_frame():
    app = _make_app()

    async def original():
        return "original"

    async def swapped():
        return "swapped"

    app.dependency_overrides[original] = swapped

    async def pusher():
        with policy_frame():
            return None

    @app.get("/")
    async def root(
        _: None = Depends(pusher),
        value: str = Depends(original),
    ):
        return {"value": value}

    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert response.json() == {"value": "swapped"}


# ---------------------------------------------------------------------------
# serialize_gate
# ---------------------------------------------------------------------------


def test_force_raw_denies_dict_response_and_does_not_fake_bytes():
    app = _make_app()

    async def gate():
        push_policy(serialize_gate=SerializePolicy.FORCE_RAW)
        return None

    @app.get("/")
    async def root(_: None = Depends(gate)):
        return {"hello": "world"}

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/")
    assert response.status_code == 503
    assert "serialize_gate:force_raw" in response.text
    # No body was ever produced for the denied serialization.
    assert response.text != '{"hello":"world"}'


def test_force_raw_keeps_callers_object_and_restores_on_pop():
    payload = {"x": [1, 2, 3], "nested": {"y": 4}}
    from fastapi.encoders import jsonable_encoder

    with pytest.raises(SerializationDenied) as exc_info:
        with policy_frame(serialize_gate=SerializePolicy.FORCE_RAW):
            jsonable_encoder(payload)
    assert exc_info.value.kind == "jsonable_encoder"
    # The caller still owns the intact original object.
    assert payload == {"x": [1, 2, 3], "nested": {"y": 4}}
    # Pop restored the empty stack: serialization works again.
    assert get_policy_view().depth == 0
    assert jsonable_encoder(payload) == payload


def test_deny_new_jsonable_encoder_direct():
    with pytest.raises(SerializationDenied) as exc_info:
        with policy_frame(serialize_gate=SerializePolicy.DENY_NEW):
            from fastapi.encoders import jsonable_encoder

            jsonable_encoder({"a": 1})
    assert exc_info.value.kind == "jsonable_encoder"


def test_json_render_gates_directly():
    with pytest.raises(SerializationDenied) as exc_info:
        with policy_frame(serialize_gate=SerializePolicy.DENY_NEW):
            StarletteJSONResponse({"a": 1})
    assert exc_info.value.kind == "json_render"


def test_orjson_render_gate():
    pytest.importorskip("orjson")
    from fastapi.responses import ORJSONResponse

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FastAPIDeprecationWarning)
        with pytest.raises(SerializationDenied) as exc_info:
            with policy_frame(serialize_gate=SerializePolicy.FORCE_RAW):
                ORJSONResponse({"a": 1})
    assert exc_info.value.kind == "orjson_render"


def test_ujson_render_gate():
    pytest.importorskip("ujson")
    from fastapi.responses import UJSONResponse

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FastAPIDeprecationWarning)
        with pytest.raises(SerializationDenied) as exc_info:
            with policy_frame(serialize_gate=SerializePolicy.DENY_NEW):
                UJSONResponse({"a": 1})
    assert exc_info.value.kind == "ujson_render"


def test_response_model_serialization_path_is_gated():
    from pydantic import BaseModel

    class Item(BaseModel):
        name: str

    app = _make_app()

    async def gate():
        push_policy(serialize_gate=SerializePolicy.DENY_NEW)

    @app.get("/", response_model=Item)
    async def root(_: None = Depends(gate)):
        return Item(name="hammer")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/")
    assert response.status_code == 503
    assert "serialize_gate:deny_new" in response.text


def test_user_returned_json_response_render_is_gated():
    app = _make_app()

    async def gate():
        push_policy(serialize_gate=SerializePolicy.FORCE_RAW)

    @app.get("/")
    async def root(_: None = Depends(gate)):
        return StarletteJSONResponse({"built": "in-endpoint"})

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/")
    assert response.status_code == 503


def test_in_flight_encoding_completes_when_frame_pushed_midwalk():
    from fastapi.encoders import jsonable_encoder

    pushed = {"done": False}

    class Holder:
        def __init__(self, value):
            self.value = value

    def custom_encode(obj: Holder):
        if "frame" not in pushed:
            # Frame pushed mid-walk, while the first encoding is in flight.
            pushed["frame"] = push_policy(serialize_gate=SerializePolicy.DENY_NEW)
        # The walk that already entered must finish with real data.
        return {"value": obj.value, "double": obj.value * 2}

    payload = {"first": Holder(21), "second": [Holder(1), Holder(2)]}
    result = jsonable_encoder(payload, custom_encoder={Holder: custom_encode})
    # Complete, non-empty result for the in-flight encoding.
    assert result == {
        "first": {"value": 21, "double": 42},
        "second": [{"value": 1, "double": 2}, {"value": 2, "double": 4}],
    }
    with pytest.raises(SerializationDenied):
        jsonable_encoder({"new": True})
    pop_policy(pushed["frame"])
    assert jsonable_encoder({"new": True}) == {"new": True}


def test_raw_streaming_body_is_not_a_json_render():
    from starlette.responses import StreamingResponse

    app = _make_app()

    async def gate():
        # force_raw stays active while the body is streamed.
        push_policy(serialize_gate=SerializePolicy.FORCE_RAW)

    @app.get("/")
    async def root(_: None = Depends(gate)):
        async def chunks():
            yield b"chunk-1\n"
            await asyncio.sleep(0)
            yield b"chunk-2\n"

        return StreamingResponse(chunks(), media_type="application/octet-stream")

    client = TestClient(app, raise_server_exceptions=False)
    with client.stream("GET", "/") as response:
        assert response.status_code == 200
        body = b"".join(response.iter_bytes())
    assert body == b"chunk-1\nchunk-2\n"


async def _jsonl_stream_case():
    # Plain app: after response.start a handled error cannot be converted,
    # so the gate error must surface directly on the stream.
    app = FastAPI()

    async def gate():
        push_policy(serialize_gate=SerializePolicy.DENY_NEW)

    @app.get("/")
    async def root(_: None = Depends(gate)):
        yield {"n": 1}
        yield {"n": 2}  # pragma: no cover - first item already fails

    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://t") as client:
            await client.get("/")
        return None
    except BaseException as exc:
        return exc


def test_jsonl_stream_gates_each_new_item_serialization():
    # ASGITransport drives the whole app call before returning, so the gate
    # error raised by the first item surfaces at request entry; the key
    # contract is the observable failure (never a faked complete body).
    exc = _run(_jsonl_stream_case())
    assert isinstance(exc, SerializationDenied)
    assert exc.kind == "stream_item"


def _flatten(exc: BaseException) -> list[BaseException]:
    errors = [exc]
    for nested in getattr(exc, "exceptions", ()):
        errors.extend(_flatten(nested))
    return errors


async def _sse_stream_case():
    from fastapi.sse import EventSourceResponse, ServerSentEvent

    # Plain app (see _jsonl_stream_case): post-start failures surface raw.
    app = FastAPI()

    async def gate():
        push_policy(serialize_gate=SerializePolicy.DENY_NEW)

    @app.get("/", response_class=EventSourceResponse)
    async def stream(_: None = Depends(gate)):
        yield ServerSentEvent(raw_data="raw-ok")
        # Let the raw event flush to the wire before the next event is
        # serialized, mirroring an idle SSE producer.
        await asyncio.sleep(0.1)
        yield ServerSentEvent(data={"must": "serialize"})  # gated

    sent_bodies: list[bytes] = []

    async def recording_app(scope, receive, send):  # type: ignore[no-untyped-def]
        async def wrapped_send(message):
            if message.get("type") == "http.response.body":
                sent_bodies.append(bytes(message.get("body", b"")))
            await send(message)

        await app(scope, receive, wrapped_send)

    transport = ASGITransport(app=recording_app)
    try:
        async with AsyncClient(transport=transport, base_url="http://t") as client:
            await client.get("/")
        return b"".join(sent_bodies), None
    except BaseException as exc:
        return b"".join(sent_bodies), exc


def test_sse_raw_events_flow_while_serialized_events_are_refused():
    body, exc = _run(_sse_stream_case())
    # The raw (non-serialized) event was delivered before the failure...
    assert b"raw-ok" in body
    # ...and the serialized event ended the stream observably (the anyio
    # producer task group wraps the gate error in an ExceptionGroup).
    assert exc is not None
    assert any(isinstance(item, SerializationDenied) for item in _flatten(exc))
    assert b'"must"' not in body  # the gated event never reached the wire


# ---------------------------------------------------------------------------
# background_gate
# ---------------------------------------------------------------------------


def test_deny_new_blocks_add_task_observably():
    app = _make_app()
    ran = []

    def work():  # pragma: no cover - must never be marked enqueued
        ran.append("ran")

    async def gate():
        push_policy(background_gate=BackgroundPolicy.DENY_NEW)

    @app.get("/")
    async def root(tasks: BackgroundTasks, _: None = Depends(gate)):
        with pytest.raises(BackgroundEnqueueDenied):
            tasks.add_task(work)
        return {"ok": True}

    # The deny surfaces at the add_task call; the handler above catches it to
    # assert the exact type, so request still completes, and no task exists.
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert ran == []


def test_deny_new_uncaught_surfaces_as_axis_error():
    app = _make_app()

    def work():  # pragma: no cover
        pass

    async def gate():
        push_policy(background_gate=BackgroundPolicy.DENY_NEW)

    @app.get("/")
    async def root(tasks: BackgroundTasks, _: None = Depends(gate)):
        tasks.add_task(work)
        return {"ok": True}

    response = TestClient(app, raise_server_exceptions=False).get("/")
    assert response.status_code == 503
    assert "background_gate:deny_new" in response.text


def test_force_cancel_withdraws_pending_tasks_on_pop():
    ran = []
    cancelled_records = []

    def work(name):
        ran.append(name)

    tasks = BackgroundTasks()
    frame = push_policy(
        background_gate=BackgroundPolicy.FORCE_CANCEL,
        on_background_cancel=cancelled_records.append,
    )
    tasks.add_task(work, "a")
    tasks.add_task(work, "b")
    popped = pop_policy(frame)

    assert ran == []  # withdrawal happens before dispatch
    assert len(popped.cancelled_tasks) == 2
    names = {record.func_name for record in popped.cancelled_tasks}
    assert names == {"test_force_cancel_withdraws_pending_tasks_on_pop.<locals>.work"}
    assert cancelled_records == list(popped.cancelled_tasks)
    assert all(
        isinstance(record, CancelledBackgroundTask) for record in cancelled_records
    )
    assert {record.reason for record in popped.cancelled_tasks} == {"force_cancel"}

    # Dispatch observably skips both tasks.
    _run(tasks())
    assert ran == []


def test_force_cancel_through_endpoint_skips_dispatch():
    app = _make_app()
    ran = []
    captured: dict[str, RequestPolicyFrame] = {}

    def work(name):
        ran.append(name)

    @app.get("/")
    async def root(tasks: BackgroundTasks):
        frame = push_policy(background_gate=BackgroundPolicy.FORCE_CANCEL)
        tasks.add_task(work, "in-frame")
        captured["frame"] = pop_policy(frame)
        return {"ok": True}

    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert ran == []
    assert len(captured["frame"].cancelled_tasks) == 1


def test_force_cancel_keeps_tasks_enqueued_before_the_frame():
    ran = []

    def work(name):
        ran.append(name)

    tasks = BackgroundTasks()
    tasks.add_task(work, "before-frame")

    frame = push_policy(background_gate=BackgroundPolicy.FORCE_CANCEL)
    tasks.add_task(work, "inside-frame")
    pop_policy(frame)

    _run(tasks())
    assert ran == ["before-frame"]


def test_force_cancel_abnormal_pop_still_withdraws():
    ran = []

    def work():
        ran.append("ran")

    tasks = BackgroundTasks()
    with pytest.raises(ValueError, match="boom"):
        with policy_frame(background_gate=BackgroundPolicy.FORCE_CANCEL) as frame:
            tasks.add_task(work)
            raise ValueError("boom")
    assert ran == []
    assert len(frame.cancelled_tasks) == 1


def test_force_cancel_during_dispatch_cancels_late_enqueued_task():
    ran = []
    late_frame: dict[str, RequestPolicyFrame] = {}

    def late_task():  # pragma: no cover - withdrawn before its turn
        ran.append("late")

    def first_task(tasks: BackgroundTasks):
        ran.append("first")
        frame = push_policy(background_gate=BackgroundPolicy.FORCE_CANCEL)
        late_frame["frame"] = frame
        # Enqueue while final dispatch iteration is already running.
        tasks.add_task(late_task)
        pop_policy(frame)

    tasks = BackgroundTasks()
    tasks.add_task(first_task, tasks)
    _run(tasks())
    assert ran == ["first"]
    assert len(late_frame["frame"].cancelled_tasks) == 1


def test_deny_new_during_dispatch_refuses_new_enqueue():
    tasks = BackgroundTasks()

    def first(tasks: BackgroundTasks):
        with policy_frame(background_gate=BackgroundPolicy.DENY_NEW):
            with pytest.raises(BackgroundEnqueueDenied):
                tasks.add_task(lambda: None)

    tasks.add_task(first, tasks)
    _run(tasks())  # no error escapes; the refusal was observable


def test_force_cancel_protects_running_task_but_cancels_pending_sibling():
    ran = []
    state: dict[str, object] = {}

    async def later():  # pragma: no cover - withdrawn before its turn
        ran.append("later")

    async def holder(tasks: BackgroundTasks):
        # This task is already in flight when the frame pops: it must finish.
        ran.append("holder-started")
        await asyncio.sleep(0)
        pop_policy(state["frame"])  # type: ignore[arg-type]
        ran.append("holder-finished")

    async def scenario():
        tasks = BackgroundTasks()
        # Push and pop from the same cooperative scope (the request task).
        frame = push_policy(background_gate=BackgroundPolicy.FORCE_CANCEL)
        state["frame"] = frame
        tasks.add_task(holder, tasks)
        tasks.add_task(later)
        await tasks()
        return frame

    frame = _run(scenario())

    # The in-flight task ran to completion; its pending sibling was withdrawn.
    assert ran == ["holder-started", "holder-finished"]
    assert len(frame.cancelled_tasks) == 1
    assert frame.cancelled_tasks[0].func_name.endswith("later")


# ---------------------------------------------------------------------------
# Context manager / stack mechanics
# ---------------------------------------------------------------------------


def test_policy_frame_pops_on_exception_and_restores_outer():
    outer = push_policy(serialize_gate=SerializePolicy.DENY_NEW)
    with pytest.raises(RuntimeError, match="x"):
        with policy_frame(dep_admit=DepAdmitPolicy.DENY_NEW):
            assert get_policy_view().depth == 2
            raise RuntimeError("x")
    assert get_policy_view().depth == 1
    with pytest.raises(SerializationDenied):
        admit_serialization("anything")
    pop_policy(outer)
    assert get_policy_view().depth == 0
    admit_serialization("anything")  # empty stack: pass-through


def test_pop_guardrails():
    with pytest.raises(RuntimeError):
        pop_policy()
    first = push_policy()
    second = push_policy()
    with pytest.raises(RuntimeError):
        pop_policy(first)  # LIFO violation
    pop_policy(second)
    pop_policy(first)


# ---------------------------------------------------------------------------
# Per-request isolation on a single event loop
# ---------------------------------------------------------------------------


async def _concurrent_dep_case():
    app = _make_app()
    b_seen = asyncio.Event()
    effects = {"a": [], "b": []}

    async def dep_a_blocking():
        effects["a"].append("enter")
        push_policy(dep_admit=DepAdmitPolicy.DENY_NEW)
        await b_seen.wait()
        effects["a"].append("released")

    async def dep_a_late():  # pragma: no cover - denied by A's own frame
        effects["a"].append("late")

    async def dep_b():
        effects["b"].append(f"depth:{get_policy_view().depth}")
        b_seen.set()
        return "b-value"

    @app.get("/a")
    async def route_a(
        blocking: None = Depends(dep_a_blocking),
        late: None = Depends(dep_a_late),
    ):
        return {"a": True}

    @app.get("/b")
    async def route_b(value: str = Depends(dep_b)):
        return {"value": value}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        task_a = asyncio.create_task(client.get("/a"))
        await asyncio.sleep(0.05)
        task_b = asyncio.create_task(client.get("/b"))
        rb = await task_b
        ra = await task_a
    return ra, rb, effects


def test_concurrent_requests_do_not_share_dep_stack():
    ra, rb, effects = _run(_concurrent_dep_case())
    assert rb.status_code == 200
    assert rb.json() == {"value": "b-value"}
    assert ra.status_code == 503
    assert "dep_admit:deny_new" in ra.text
    assert effects["b"] == ["depth:0"]
    assert "late" not in effects["a"]


async def _concurrent_serialize_case():
    app = _make_app()
    b_seen = asyncio.Event()

    async def gate_a():
        push_policy(serialize_gate=SerializePolicy.FORCE_RAW)
        await b_seen.wait()

    async def gate_b():
        b_seen.set()
        return None

    @app.get("/a")
    async def route_a(_: None = Depends(gate_a)):
        return {"who": "a"}

    @app.get("/b")
    async def route_b(_: None = Depends(gate_b)):
        return {"who": "b"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        ta = asyncio.create_task(client.get("/a"))
        await asyncio.sleep(0.05)
        tb = asyncio.create_task(client.get("/b"))
        rb = await tb
        ra = await ta
    return ra, rb


def test_concurrent_serialize_gates_do_not_cross():
    ra, rb = _run(_concurrent_serialize_case())
    assert rb.status_code == 200
    assert rb.json() == {"who": "b"}
    assert ra.status_code == 503
    assert "force_raw" in ra.text


async def _concurrent_background_case():
    app = _make_app()
    b_seen = asyncio.Event()
    ran = {"a": [], "b": []}

    def task(who):
        ran[who].append("task")

    async def gate_a():
        frame = push_policy(background_gate=BackgroundPolicy.FORCE_CANCEL)
        await b_seen.wait()
        return frame

    async def gate_b():
        b_seen.set()
        return None

    @app.get("/a")
    async def route_a(
        tasks: BackgroundTasks,
        frame: RequestPolicyFrame = Depends(gate_a),
    ):
        tasks.add_task(task, "a")
        pop_policy(frame)
        return {"who": "a"}

    @app.get("/b")
    async def route_b(tasks: BackgroundTasks, _: None = Depends(gate_b)):
        tasks.add_task(task, "b")
        return {"who": "b"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        ta = asyncio.create_task(client.get("/a"))
        await asyncio.sleep(0.05)
        tb = asyncio.create_task(client.get("/b"))
        rb = await tb
        ra = await ta
    return ra, rb, ran


def test_concurrent_force_cancel_does_not_hit_neighbor_request():
    ra, rb, ran = _run(_concurrent_background_case())
    assert ra.status_code == 200
    assert rb.status_code == 200
    assert ran == {"a": [], "b": ["task"]}
