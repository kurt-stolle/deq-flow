import torch

from ._ops import batch_masked_mixing, line_search, matvec, rmatvec
from ._stats import init_solver_stats
from .typings import Func, Solution, StopMode


def anderson(
    func: Func,
    x0: torch.Tensor,
    threshold: int,
    *,
    eps=1e-3,
    stop_mode=StopMode.REL,
    indexing=None,
    m=6,
    lam=1e-4,
    beta=1.0,
) -> Solution:
    """Anderson acceleration for fixed point iteration."""
    stop_mode = stop_mode.value
    bsz, dim = x0.flatten(start_dim=1).shape
    f = lambda x: func(x.view_as(x0)).view_as(x)

    alternative_mode = "rel" if stop_mode == "abs" else "abs"
    X = torch.zeros(bsz, m, dim, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, dim, dtype=x0.dtype, device=x0.device)

    x0_flat = x0.flatten(start_dim=1)
    X[:, 0], F[:, 0] = x0_flat, f(x0_flat)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0])

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    trace_dict, lowest_dict, lowest_step_dict = init_solver_stats(x0)
    lowest_xest = x0

    indexing_list = []

    for k in range(2, threshold):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1 : n + 1, 1 : n + 1] = (
            torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        )
        alpha = torch.linalg.solve(H[:, : n + 1, : n + 1], y[:, : n + 1])[:, 1 : n + 1, 0]  # (bsz x n)

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m])
        gx = F[:, k % m] - X[:, k % m]
        abs_diff = gx.norm(dim=1)
        rel_diff = abs_diff / (F[:, k % m].norm(dim=1) + 1e-8)

        diff_dict = {"abs": abs_diff, "rel": rel_diff}
        trace_dict["abs"].append(abs_diff)
        trace_dict["rel"].append(rel_diff)

        for mode in ["rel", "abs"]:
            is_lowest = diff_dict[mode] < lowest_dict[mode]
            if mode == stop_mode:
                lowest_xest = batch_masked_mixing(is_lowest, F[:, k % m], lowest_xest)
                lowest_xest = lowest_xest.view_as(x0).clone().detach()
            lowest_dict[mode] = batch_masked_mixing(is_lowest, diff_dict[mode], lowest_dict[mode])
            lowest_step_dict[mode] = batch_masked_mixing(is_lowest, k + 1, lowest_step_dict[mode])

        if indexing and (k + 1) in indexing:
            indexing_list.append(lowest_xest)

        if trace_dict[stop_mode][-1].max() < eps:
            for _ in range(threshold - 1 - k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break

    # at least return the lowest value when enabling  ``indexing''
    if indexing and not indexing_list:
        indexing_list.append(lowest_xest)

    X = F = None

    info = {
        "abs_lowest": lowest_dict["abs"],
        "rel_lowest": lowest_dict["rel"],
        "abs_trace": trace_dict["abs"],
        "rel_trace": trace_dict["rel"],
        "nstep": lowest_step_dict[stop_mode],
    }
    return lowest_xest, indexing_list, info
