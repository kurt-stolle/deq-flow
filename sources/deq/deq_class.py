from __future__ import annotations

from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Any, Callable, Iterable, Optional, Sequence, TypeAlias

import numpy as np
import torch
import torch.nn as nn

from . import solvers
from .grad import Backward, BackwardIFT, BackwardPhantom, make_pair
from .jacobian import power_method

__all__ = ["DEQIndexing", "DEQSliced", "DEQBase"]


DEQInfo: TypeAlias = dict[str, torch.Tensor]
DEQResult: TypeAlias = tuple[torch.Tensor, DEQInfo]


class DEQBase(nn.Module, metaclass=ABCMeta):
    produce_grad: list[Backward]

    def __init__(
        self,
        *,
        solver: solvers.Solver,
        threshold: int,
        threshold_eval: Optional[int] = None,
    ):
        super().__init__()

        self.solver = solver

        self.threshold = threshold
        self.threshold_eval = threshold_eval or threshold

    @staticmethod
    def log_convergence_hook(mod: DEQBase, input: tuple[Any, ...], output: DEQResult) -> None:
        mod.log_convergence(output[1])

    @torch.jit.unused
    @torch.no_grad()
    def log_convergence(self, info: DEQInfo, name="FORWARD", writer: Callable[[str], None] = print) -> None:
        state = "TRAIN" if self.training else "VALID"

        rel_lowest, abs_lowest = info["rel_lowest"].mean().item(), info["abs_lowest"].mean().item()
        nstep = info["nstep"]

        show_str = f"{state} | {name} | rel: {rel_lowest}; abs: {abs_lowest}; nstep: {nstep}"
        writer(show_str)

    def _sradius(self, deq, z_star):
        with torch.enable_grad():
            new_z_star = deq(z_star.requires_grad_())
        _, sradius = power_method(new_z_star, z_star, n_iters=75)

        return sradius

    @abstractmethod
    def solve_fixed_point(self, deq: solvers.Func, z_init: torch.Tensor, threshold: int) -> DEQResult:
        ...

    @abstractmethod
    def forward_training(
        self, deq: solvers.Func, z_init: torch.Tensor, sradius_mode=False, writer=None, **kwargs
    ) -> DEQResult:
        ...

    def forward_inference(
        self, deq: solvers.Func, z_init: torch.Tensor, sradius_mode=False, writer=None, **kwargs
    ) -> DEQResult:
        """
        During inference, we directly solve for fixed point.
        """
        z_star, _, info = self.solve_fixed_point(deq, z_init, self.threshold_eval)

        sradius = self._sradius(deq, z_star) if sradius_mode else torch.zeros(1, device=z_star.device)
        info["sradius"] = sradius

        z_out = [deq.vec2list(z_star)]

        return z_out, info

    def forward(self, deq: solvers.Func, z_init: torch.Tensor, sradius_mode=False, writer=None, **kwargs) -> DEQResult:
        if self.training:
            return self.forward_training(
                deq,
                z_init,
                sradius_mode=sradius_mode,
                writer=writer,
                **kwargs,
            )
        else:
            return self.forward_inference(
                deq,
                z_init,
                sradius_mode=sradius_mode,
                writer=writer,
                **kwargs,
            )


class DEQIndexing(DEQBase):
    def __init__(
        self,
        *,
        n_losses: int = 1,
        indexing: Optional[Sequence[int]] = None,
        phantom_grad: Optional[Sequence[int]] = None,
        ift: Optional[BackwardIFT] = None,
        tau: float = 1.0,
        sup_all: float = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if indexing is None:
            indexing = []
        if phantom_grad is None:
            phantom_grad = [1]

        # Define gradient functions through the backward factory
        if n_losses > 1:
            n_losses = min(self.threshold, n_losses)
            delta = int(self.threshold // n_losses)
            self.indexing = [(k + 1) * delta for k in range(n_losses)]
        else:
            self.indexing = [*indexing, self.threshold]

        # By default, we use the same phantom grad for all corrections.
        # You can also set different grad steps a, b, and c for different terms by ``args.phantom_grad a b c ...''.
        indexing_pg = make_pair(self.indexing, phantom_grad)
        produce_grad = [BackwardPhantom(n=pg, tau=tau, sup_all=sup_all) for pg in indexing_pg]
        if ift is not None:
            produce_grad[-1] = ift

        self.produce_grad = produce_grad

    def solve_fixed_point(self, deq: solvers.Func, z_init: torch.Tensor, threshold: int) -> solvers.Solution:
        indexing = self.indexing if self.training else None

        with torch.no_grad():
            z_star, trajectory, info = self.solver(
                deq,
                x0=z_init,
                threshold=threshold,  # To reuse previous coarse fixed points
                indexing=indexing,
            )
        return z_star, trajectory, info

    def forward_training(self, deq: solvers.Func, z_init: torch.Tensor, sradius_mode=False, writer=None, **kwargs):
        _, trajectory, info = self.solve_fixed_point(deq, z_init, f_thres=self.threshold)

        z_out = []
        for z_pred, produce_grad in zip(trajectory, self.produce_grad):
            z_out += produce_grad(self, deq, z_pred)  # See lib/grad.py for the backward pass implementations

        z_out = [deq.vec2list(each) for each in z_out]

        return z_out, info


class DEQSliced(DEQBase):
    def __init__(
        self,
        *,
        n_losses: int = 1,
        indexing: Optional[Sequence[int]] = None,
        phantom_grad: Optional[Sequence[int]] = None,
        ift: Optional[BackwardIFT] = None,
        tau: float = 1.0,
        sup_all: float = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if indexing is None:
            indexing = []
        if phantom_grad is None:
            phantom_grad = [1]

        # Define gradient functions through the backward factory
        if n_losses > 1:
            self.indexing = [int(self.threshold // n_losses) for _ in range(n_losses)]
        else:
            self.indexing = np.diff([0, *indexing, self.threshold]).tolist()

        # By default, we use the same phantom grad for all corrections.
        # You can also set different grad steps a, b, and c for different terms by ``args.phantom_grad a b c ...''.
        indexing_pg = make_pair(self.indexing, phantom_grad)
        produce_grad = [BackwardPhantom(n=pg, tau=tau, sup_all=sup_all) for pg in indexing_pg]
        if ift is not None:
            produce_grad[-1] = ift

        self.produce_grad = produce_grad

    def solve_fixed_point(self, deq: solvers.Func, z_init: torch.Tensor, f_thres: torch.Tensor):
        with torch.no_grad():
            z_star, trajectory, info = self.solver(
                deq,
                z_init,
                f_thres,  # To reuse previous coarse fixed points
            )
        return z_star, trajectory, info

    def forward_training(self, deq, z_star, sradius_mode=False, writer=None, **kwargs):
        z_out = []
        for f_thres, produce_grad in zip(self.indexing, self.produce_grad):
            z_star, _, info = self.solve_fixed_point(deq, z_star, f_thres=f_thres)
            z_out += produce_grad(self, deq, z_star, writer=writer)  # See lib/grad.py for implementations
            z_star = z_out[-1]  # Add the gradient chain to the solver.

        z_out = [deq.vec2list(each) for each in z_out]

        return z_star, info
