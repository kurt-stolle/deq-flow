from __future__ import annotations

import torch


def init_solver_stats(x0, init_loss=1e8):
    trace_dict = {
        "abs": [torch.tensor(init_loss, device=x0.device).repeat(x0.shape[0])],
        "rel": [torch.tensor(init_loss, device=x0.device).repeat(x0.shape[0])],
    }
    lowest_dict = {
        "abs": torch.tensor(init_loss, device=x0.device).repeat(x0.shape[0]),
        "rel": torch.tensor(init_loss, device=x0.device).repeat(x0.shape[0]),
    }
    lowest_step_dict = {
        "abs": torch.tensor(0, device=x0.device).repeat(x0.shape[0]),
        "rel": torch.tensor(0, device=x0.device).repeat(x0.shape[0]),
    }

    return trace_dict, lowest_dict, lowest_step_dict
