from abc import ABCMeta, abstractmethod
from typing import Iterable, Optional, overload

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored

from .grad import backward_factory, make_pair
from .jacobian import power_method

__all__ = ["DEQIndexing", "DEQSliced", "DEQBase"]


class DEQBase(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        *,
        f_solver: nn.Module,
        f_thres: int,
        f_eps: float,
        f_stop_mode: str,
        b_solver: nn.Module,
        b_thres: int,
        b_eps: float,
        b_stop_mode: str,
        eval_f_thres: int,
    ):
        super().__init__()

        self.f_solver = f_solver
        self.f_thres = f_thres
        self.f_eps = f_eps
        self.f_stop_mode = f_stop_mode

        self.b_solver = b_solver
        self.b_thres = b_thres
        self.b_eps = b_eps
        self.b_stop_mode = b_stop_mode

        self.eval_f_thres = eval_f_thres

        self.hook = None

    @classmethod
    def from_eval_factor(cls, eval_factor: float, *, f_thres: int, **kwargs):
        if "eval_f_thres" in kwargs:
            raise ValueError("`eval_f_thres` should not be specified as an argument when computed using `eval_factor`.")
        return cls(eval_f_thres=int(eval_factor * f_thres), **kwargs)

    def _log_convergence(self, info, name="FORWARD", color="yellow"):
        state = "TRAIN" if self.training else "VALID"
        alt_mode = "rel" if self.f_stop_mode == "abs" else "abs"

        rel_lowest, abs_lowest = info["rel_lowest"].mean().item(), info["abs_lowest"].mean().item()
        nstep = info["nstep"]

        show_str = f"{state} | {name} | rel: {rel_lowest}; abs: {abs_lowest}; nstep: {nstep}"
        print(colored(show_str, color))

    def _sradius(self, deq_func, z_star):
        with torch.enable_grad():
            new_z_star = deq_func(z_star.requires_grad_())
        _, sradius = power_method(new_z_star, z_star, n_iters=75)

        return sradius

    @abstractmethod
    def _solve_fixed_point(self, deq_func, z_init, log=False, f_thres=None, **kwargs):
        ...

    @abstractmethod
    def forward(self, deq_func, z_init, log=False, sradius_mode=False, writer=None, **kwargs):
        ...


class DEQIndexing(DEQBase):
    def __init__(
        self,
        *,
        n_losses: int,
        indexing: Iterable[float],
        phantom_grad: int = 1,
        safe_ift: Optional[bool],
        tau: float,
        sup_all: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Define gradient functions through the backward factory
        if n_losses > 1:
            n_losses = min(self.f_thres, n_losses)
            delta = int(self.f_thres // n_losses)
            self.indexing = [(k + 1) * delta for k in range(n_losses)]
        else:
            self.indexing = [*indexing, self.f_thres]

        # By default, we use the same phantom grad for all corrections.
        # You can also set different grad steps a, b, and c for different terms by ``args.phantom_grad a b c ...''.
        indexing_pg = make_pair(self.indexing, phantom_grad)
        produce_grad = [backward_factory(grad_type=pg, tau=tau, sup_all=sup_all) for pg in indexing_pg]
        if safe_ift is not None:
            # Enabling args.ift will replace the last gradient function by IFT.
            produce_grad[-1] = backward_factory(
                grad_type="ift",
                safe_ift=safe_ift,
                b_solver=self.b_solver,
                b_solver_kwargs=dict(threshold=self.b_thres, eps=self.b_eps, stop_mode=self.b_stop_mode),
            )

        self.produce_grad = produce_grad

    def _solve_fixed_point(self, deq_func, z_init, log=False, f_thres=None, **kwargs):
        if f_thres is None:
            f_thres = self.f_thres
        indexing = self.indexing if self.training else None

        with torch.no_grad():
            z_star, trajectory, info = self.f_solver(
                deq_func,
                x0=z_init,
                threshold=f_thres,  # To reuse previous coarse fixed points
                eps=self.f_eps,
                stop_mode=self.f_stop_mode,
                indexing=indexing,
            )

        if log:
            self._log_convergence(info, name="FORWARD", color="yellow")

        return z_star, trajectory, info

    def forward(self, deq_func, z_init, log=False, sradius_mode=False, writer=None, **kwargs):
        if self.training:
            _, trajectory, info = self._solve_fixed_point(deq_func, z_init, log=log, *kwargs)

            z_out = []
            for z_pred, produce_grad in zip(trajectory, self.produce_grad):
                z_out += produce_grad(self, deq_func, z_pred)  # See lib/grad.py for the backward pass implementations

            z_out = [deq_func.vec2list(each) for each in z_out]
        else:
            # During inference, we directly solve for fixed point
            z_star, _, info = self._solve_fixed_point(deq_func, z_init, log=log, f_thres=self.eval_f_thres)

            sradius = self._sradius(deq_func, z_star) if sradius_mode else torch.zeros(1, device=z_star.device)
            info["sradius"] = sradius

            z_out = [deq_func.vec2list(z_star)]

        return z_out, info


class DEQSliced(DEQBase):
    def __init__(
        self,
        *,
        indexing: Iterable[int],
        n_losses: int,
        sup_all: bool,
        safe_ift: Optional[bool],
        tau: float = 1,
        phantom_grad: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Define gradient functions through the backward factory
        if n_losses > 1:
            self.indexing = [int(self.f_thres // self.n_losses) for _ in range(n_losses)]
        else:
            self.indexing = np.diff([0, *indexing, self.f_thres]).tolist()

        # By default, we use the same phantom grad for all corrections.
        # You can also set different grad steps a, b, and c for different terms by ``args.phantom_grad a b c ...''.
        indexing_pg = make_pair(self.indexing, phantom_grad)
        produce_grad = [backward_factory(grad_type=pg, tau=tau, sup_all=sup_all) for pg in indexing_pg]
        if safe_ift is not None:
            # Enabling args.ift will replace the last gradient function by IFT.
            produce_grad[-1] = backward_factory(
                grad_type="ift",
                safe_ift=safe_ift,
                b_solver=self.b_solver,
                b_solver_kwargs=dict(threshold=args.b_thres, eps=args.b_eps, stop_mode=args.b_stop_mode),
            )

        self.produce_grad = produce_grad

    def _solve_fixed_point(self, deq_func, z_init, log=False, f_thres=None, **kwargs):
        with torch.no_grad():
            z_star, _, info = self.f_solver(
                deq_func,
                x0=z_init,
                threshold=f_thres,  # To reuse previous coarse fixed points
                eps=self.f_eps,
                stop_mode=self.f_stop_mode,
            )

        if log:
            self._log_convergence(info, name="FORWARD", color="yellow")

        return z_star, info

    def forward(self, deq_func, z_star, log=False, sradius_mode=False, writer=None, **kwargs):
        if self.training:
            z_out = []
            for f_thres, produce_grad in zip(self.indexing, self.produce_grad):
                z_star, info = self._solve_fixed_point(deq_func, z_star, f_thres=f_thres, log=log)
                z_out += produce_grad(self, deq_func, z_star, writer=writer)  # See lib/grad.py for implementations
                z_star = z_out[-1]  # Add the gradient chain to the solver.

            z_out = [deq_func.vec2list(each) for each in z_out]
        else:
            # During inference, we directly solve for fixed point
            z_star, info = self._solve_fixed_point(deq_func, z_star, f_thres=self.eval_f_thres, log=log)

            sradius = self._sradius(deq_func, z_star) if sradius_mode else torch.zeros(1, device=z_star.device)
            info["sradius"] = sradius

            z_out = [deq_func.vec2list(z_star)]

        return z_out, info
