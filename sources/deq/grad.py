from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Literal, Optional, TypeAlias

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.utils import hooks

from . import solvers


def make_pair(target, source):
    if len(target) == len(source):
        return source
    elif len(source) == 1:
        return [source[0] for _ in range(len(target))]
    else:
        raise ValueError("Unable to align the arg squence!")


class Backward(ABC):
    @abstractmethod
    def __call__(self, f: solvers.Func, z_pred: torch.Tensor) -> list[torch.Tensor]:
        ...


class BackwardIFT(Backward):
    __slots__ = ["_solver", "_threshold", "_hook"]

    _solver: solvers.Solver
    _threshold: int
    _hook: hooks.RemovableHandle | None

    def __new__(cls, solver=solvers.anderson, threshold=50):
        self = super().__new__(cls)
        self._solver = solver
        self._threshold = threshold
        self._hook = None

        return self

    def __call__(self, f: solvers.Func, z_pred: torch.Tensor) -> list[torch.Tensor]:
        z_pred = z_pred.requires_grad_()
        new_z_pred = f(z_pred)  # 1-step grad for df/dtheta

        def backward_hook(grad):
            if self._hook is not None:
                self._hook.remove()  # To avoid infinite loop
            grad_star, _, info = self._solver(
                lambda y: autograd.grad(new_z_pred, z_pred, y, retain_graph=True)[0] + grad,
                torch.zeros_like(grad),
                self._threshold,
            )
            return grad_star

        self._hook = new_z_pred.register_hook(backward_hook)

        return [new_z_pred]


class BackwardPhantom(Backward):
    def __new__(cls, n: int = 1, sup_all: bool = False, tau: float = 1.0):
        if sup_all:
            def sup_all_phantom_grad(func, z_pred):
                z_out = []
                for _ in range(n):
                    z_pred = (1 - tau) * z_pred + tau * func(z_pred)
                    z_out.append(z_pred)

                return z_out

            return sup_all_phantom_grad
        else:

            def phantom_grad(func, z_pred):
                for _ in range(n):
                    z_pred = (1 - tau) * z_pred + tau * func(z_pred)

                return [z_pred]

            return phantom_grad
