from collections.abc import Callable, Sequence
from typing import Annotated, Any

from annotated_doc import Doc
from fastapi.dependency_policy import (
    POLICY_STACK_SCOPE_KEY,
    TaskAdmit,
    TaskRefusedError,
    _PolicyStackState,
)
from fastapi.telemetry._api import _operation
from starlette.background import BackgroundTask
from starlette.background import BackgroundTasks as StarletteBackgroundTasks
from starlette.types import Scope
from typing_extensions import ParamSpec

P = ParamSpec("P")


class BackgroundTasks(StarletteBackgroundTasks):
    """
    A collection of background tasks that will be called after a response has been
    sent to the client.

    Read more about it in the
    [FastAPI docs for Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/).

    ## Example

    ```python
    from fastapi import BackgroundTasks, FastAPI

    app = FastAPI()


    def write_notification(email: str, message=""):
        with open("log.txt", mode="w") as email_file:
            content = f"notification for {email}: {message}"
            email_file.write(content)


    @app.post("/send-notification/{email}")
    async def send_notification(email: str, background_tasks: BackgroundTasks):
        background_tasks.add_task(write_notification, email, message="some notification")
        return {"message": "Notification sent in the background"}
    ```
    """

    def __init__(self, tasks: Sequence[BackgroundTask] | None = None) -> None:
        super().__init__(tasks)
        # ASGI scope this request-injected instance belongs to. ``None`` means
        # the instance was created outside of dependency solving, in which case
        # policy frames do not apply.
        self._fastapi_policy_scope: Scope | None = None
        # Task objects admitted under a ``mute_after_response`` frame. They stay
        # in ``self.tasks`` (observable) but are skipped after the response.
        self._fastapi_muted_tasks: set[int] = set()

    def _fastapi_bind_policy_scope(self, scope: Scope) -> None:
        self._fastapi_policy_scope = scope

    def _fastapi_policy_muted(self) -> set[int]:
        return self._fastapi_muted_tasks

    def add_task(
        self,
        func: Annotated[
            Callable[P, Any],
            Doc(
                """
                The function to call after the response is sent.

                It can be a regular `def` function or an `async def` function.
                """
            ),
        ],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """
        Add a function to be called in the background after the response is sent.

        Read more about it in the
        [FastAPI docs for Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/).
        """
        state: _PolicyStackState | None = None
        frame = None
        scope = self._fastapi_policy_scope
        if scope is not None:
            maybe_state = scope.get(POLICY_STACK_SCOPE_KEY)
            if isinstance(maybe_state, _PolicyStackState):
                state = maybe_state
                frame = state.top_frame()
                if frame is not None and frame.task_admit is TaskAdmit.REFUSE_ENQUEUE:
                    # Refusal happens before the task exists; previously queued
                    # tasks are untouched.
                    raise TaskRefusedError(
                        "BackgroundTasks.add_task() was refused by the active"
                        " 'refuse_enqueue' dependency policy frame"
                    )
        super().add_task(func, *args, **kwargs)
        if state is not None:
            # Count every admission so a later 'refuse_enqueue' push can be
            # rejected instead of pretending earlier tasks never existed.
            state.admitted_tasks += 1
            if frame is not None and frame.task_admit is TaskAdmit.MUTE_AFTER_RESPONSE:
                self._fastapi_muted_tasks.add(id(self.tasks[-1]))

    async def __call__(self) -> None:
        for task in self.tasks:
            # Muted tasks remain in the queue, they are simply never executed;
            # the queue is deliberately not cleared to fake the behavior.
            if id(task) in self._fastapi_muted_tasks:
                continue
            with _operation(name="background_task", function=task.func):
                await task()
