from enum import Enum
from typing import Callable, TypeAlias

import torch

Solution: TypeAlias = tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]
Func: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
# Solver: Callable[[Func, torch.Tensor, int], Solution]

def Solver(
    f: Func, x0: torch.Tensor, threshold, **kwargs
) -> Solution:
    raise NotImplementedError("Solver is an abstract type.")

class StopMode(Enum):
    ABS = "abs"
    REL = "rel"
