import torch

from ._ops import batch_masked_mixing, line_search, matvec, rmatvec
from ._stats import init_solver_stats
from .typings import Func, Solution, StopMode


def naive_solver(
    f: Func, x0: torch.Tensor, threshold: int, *, eps=1e-3, stop_mode=StopMode.REL, indexing=None, return_final=True
) -> Solution:
    """Naive Unrolling for fixed point iteration."""
    stop_mode = stop_mode.value
    alternative_mode = "rel" if stop_mode == "abs" else "abs"

    trace_dict, lowest_dict, lowest_step_dict = init_solver_stats(x0)
    lowest_xest = x0

    indexing_list = []

    fx = x = x0
    for k in range(threshold):
        x = fx
        fx = f(x)
        gx = fx - x
        abs_diff = gx.flatten(start_dim=1).norm(dim=1)
        rel_diff = abs_diff / (fx.flatten(start_dim=1).norm(dim=1) + 1e-8)

        diff_dict = {"abs": abs_diff, "rel": rel_diff}
        trace_dict["abs"].append(abs_diff)
        trace_dict["rel"].append(rel_diff)

        for mode in ["rel", "abs"]:
            is_lowest = diff_dict[mode] < lowest_dict[mode] + return_final
            if return_final and mode == stop_mode:
                lowest_xest = batch_masked_mixing(is_lowest, fx, lowest_xest)
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

    info = {
        "abs_lowest": lowest_dict["abs"],
        "rel_lowest": lowest_dict["rel"],
        "abs_trace": trace_dict["abs"],
        "rel_trace": trace_dict["rel"],
        "nstep": lowest_step_dict[stop_mode],
    }
    return lowest_xest, indexing_list, info
