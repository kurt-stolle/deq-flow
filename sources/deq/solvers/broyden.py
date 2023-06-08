import torch

from ._ops import batch_masked_mixing, line_search, matvec, rmatvec
from ._stats import init_solver_stats
from .typings import Func, Solution, StopMode


def broyden(
    func: Func,
    x0: torch.Tensor,
    threshold,
    *,
    eps=1e-3,
    stop_mode: StopMode = StopMode.REL,
    indexing=None,
    LBFGS_thres=None,
    ls=False,
) -> Solution:
    """
    Broyden's method for fixed point iteration.
    """

    stop_mode = stop_mode.value
    bsz, dim = x0.flatten(start_dim=1).shape
    g = lambda y: func(y.view_as(x0)).view_as(y) - y

    alternative_mode = "rel" if stop_mode == "abs" else "abs"
    LBFGS_thres = threshold if LBFGS_thres is None else LBFGS_thres

    x_est = x0.flatten(start_dim=1)  # (B, D)
    gx = g(x_est)  # (B, D)
    nstep = 0
    tnstep = 0

    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(
        bsz, dim, LBFGS_thres, dtype=x0.dtype, device=x0.device
    )  # One can also use an L-BFGS scheme to further reduce memory
    VTs = torch.zeros(bsz, LBFGS_thres, dim, dtype=x0.dtype, device=x0.device)
    update = -matvec(Us[:, :, :nstep], VTs[:, :nstep], gx)  # Formally should be -torch.matmul(inv_jacobian (-I), gx)
    prot_break = False

    new_objective = 1e8

    trace_dict, lowest_dict, lowest_step_dict = init_solver_stats(x0)
    nstep, lowest_xest = 0, x_est

    indexing_list = []

    while nstep < threshold:
        x_est, gx, delta_x, delta_gx, ite = line_search(update, x_est, gx, g, nstep=nstep, on=ls)
        nstep += 1
        tnstep += ite + 1
        abs_diff = gx.norm(dim=1)
        rel_diff = abs_diff / ((gx + x_est).norm(dim=1) + 1e-8)

        diff_dict = {"abs": abs_diff, "rel": rel_diff}
        trace_dict["abs"].append(abs_diff)
        trace_dict["rel"].append(rel_diff)

        for mode in ["rel", "abs"]:
            is_lowest = diff_dict[mode] < lowest_dict[mode]
            if mode == stop_mode:
                lowest_xest = batch_masked_mixing(is_lowest, x_est, lowest_xest)
                lowest_xest = lowest_xest.view_as(x0).clone().detach()
            lowest_dict[mode] = batch_masked_mixing(is_lowest, diff_dict[mode], lowest_dict[mode])
            lowest_step_dict[mode] = batch_masked_mixing(is_lowest, nstep, lowest_step_dict[mode])

        if indexing and (nstep + 1) in indexing:
            indexing_list.append(lowest_xest)

        new_objective = diff_dict[stop_mode].max()
        if new_objective < eps:
            break

        if nstep > 30:
            progress = (
                torch.stack(trace_dict[stop_mode][-30:]).max(dim=1)[0]
                / torch.stack(trace_dict[stop_mode][-30:]).min(dim=1)[0]
            )
            if new_objective < 3 * eps and progress.max() < 1.3:
                # if there's hardly been any progress in the last 30 steps
                break

        part_Us, part_VTs = Us[:, :, : nstep - 1], VTs[:, : nstep - 1]
        vT = rmatvec(part_Us, part_VTs, delta_x)
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx)) / torch.einsum("bd,bd->b", vT, delta_gx)[:, None]
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[:, (nstep - 1) % LBFGS_thres] = vT
        Us[:, :, (nstep - 1) % LBFGS_thres] = u
        update = -matvec(Us[:, :, :nstep], VTs[:, :nstep], gx)

    # Fill everything up to the threshold length
    for _ in range(threshold + 1 - len(trace_dict[stop_mode])):
        trace_dict[stop_mode].append(lowest_dict[stop_mode])
        trace_dict[alternative_mode].append(lowest_dict[alternative_mode])

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
