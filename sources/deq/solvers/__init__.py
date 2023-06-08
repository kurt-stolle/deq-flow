from .anderson import *
from .broyden import *
from .naive import *
from .typings import *
from .typings import Solver


def get(key: str) -> Solver:
    match key:
        case "naive_solver" | "naive":
            from .naive import naive_solver
            return naive_solver
        case "broyden":
            from .broyden import broyden
            return broyden
        case "anderson":
            from .anderson import anderson
            return anderson
        case _:
            raise NotImplementedError(f"Unsupported solver: {key}")
