"""FastAPI framework, high performance, easy to learn, fast to code, ready for production"""

__version__ = "0.162.0"

from starlette import status as status

from .applications import FastAPI as FastAPI
from .background import BackgroundTasks as BackgroundTasks
from .datastructures import UploadFile as UploadFile
from .dependency_policy import AstackRelease as AstackRelease
from .dependency_policy import DependencyPolicyError as DependencyPolicyError
from .dependency_policy import EffectiveDependencyPolicy as EffectiveDependencyPolicy
from .dependency_policy import OverrideLatch as OverrideLatch
from .dependency_policy import SolveReuse as SolveReuse
from .dependency_policy import TaskAdmit as TaskAdmit
from .dependency_policy import TaskRefusedError as TaskRefusedError
from .dependency_policy import dependency_policy as dependency_policy
from .dependency_policy import (
    get_effective_dependency_policy as get_effective_dependency_policy,
)
from .dependency_policy import (
    pop_dependency_policy as pop_dependency_policy,
)
from .dependency_policy import (
    push_dependency_policy as push_dependency_policy,
)
from .exceptions import HTTPException as HTTPException
from .exceptions import WebSocketException as WebSocketException
from .param_functions import Body as Body
from .param_functions import Cookie as Cookie
from .param_functions import Depends as Depends
from .param_functions import File as File
from .param_functions import Form as Form
from .param_functions import Header as Header
from .param_functions import Path as Path
from .param_functions import Query as Query
from .param_functions import Security as Security
from .requests import Request as Request
from .responses import Response as Response
from .routing import APIRouter as APIRouter
from .websockets import WebSocket as WebSocket
from .websockets import WebSocketDisconnect as WebSocketDisconnect
