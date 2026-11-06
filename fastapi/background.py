from collections.abc import Callable, Sequence
from typing import Annotated, Any

from annotated_doc import Doc
from fastapi.request_policy import (
    BackgroundAdmit,
    BackgroundTaskRejectedError,
)
from fastapi.telemetry._api import _operation
from starlette.background import BackgroundTask
from starlette.background import BackgroundTasks as StarletteBackgroundTasks
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
        # Request policy state, bound by the dependency solver for the
        # per-request BackgroundTasks instance. ``None`` means the queue was
        # created outside of request solving and keeps plain Starlette
        # behavior.
        self._fastapi_policy_state: Any = None

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
        state = self._fastapi_policy_state
        # Read the live effective frame on every admission: a reject_add
        # frame pushed after tasks were already queued must not purge them,
        # it only rejects new admissions.
        admit = state.current_admit() if state is not None else None
        if admit is BackgroundAdmit.reject_add:
            raise BackgroundTaskRejectedError(
                "BackgroundTasks.add_task() was rejected by the active request "
                "policy frame (bg_admit=reject_add)."
            )
        super().add_task(func, *args, **kwargs)
        if state is not None and admit is BackgroundAdmit.suppress_run:
            # Stamp at enqueue time: the task stays in the queue (it is never
            # cleared), but it is skipped when the queue runs and remains
            # observable as not-run.
            state.mark_suppressed(self.tasks[-1])

    async def __call__(self) -> None:
        state = self._fastapi_policy_state
        for task in self.tasks:
            if state is not None and state.is_suppressed(task):
                # Suppressed tasks are skipped, not removed from the queue.
                state.record_skipped()
                continue
            with _operation(name="background_task", function=task.func):
                await task()
            if state is not None:
                state.record_executed()
