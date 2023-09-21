from __future__ import annotations

import torch


def batch_masked_mixing(mask, mask_var, orig_var):
    """
    First align the axes of mask to mask_var.

    Then mix mask_var and orig_var through the aligned mask.

    Args:
        mask: a tensor of shape (B,)
        mask_var: a tensor of shape (B, ...) for the mask to select
        orig_var: a tensor of shape (B, ...) for the reversed mask to select
    """

    if torch.is_tensor(mask_var):
        axes_to_align = len(mask_var.shape) - 1
    elif torch.is_tensor(orig_var):
        axes_to_align = len(orig_var.shape) - 1
    else:
        raise ValueError("Either mask_var or orig_var should be a Pytorch tensor!")

    aligned_mask = mask.view(mask.shape[0], *[1 for _ in range(axes_to_align)])

    return aligned_mask * mask_var + ~aligned_mask * orig_var


def _safe_norm(v):
    if not torch.isfinite(v).all():
        return torch.inf
    return torch.norm(v)


def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    ite = 0
    phi_a0 = phi(alpha0)  # First do an update with step size 1
    if phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        return alpha0, phi_a0, ite

    # Otherwise, compute the minimizer of a quadratic interpolant
    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    while alpha1 > amin:  # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1 - alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0 * alpha1) - alpha1**2 * (phi_a0 - phi0 - derphi0 * alpha0)
        a = a / factor
        b = -(alpha0**3) * (phi_a1 - phi0 - derphi0 * alpha1) + alpha1**3 * (phi_a0 - phi0 - derphi0 * alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * derphi0))) / (3.0 * a)
        phi_a2 = phi(alpha2)
        ite += 1

        if phi_a2 <= phi0 + c1 * alpha2 * derphi0:
            return alpha2, phi_a2, ite

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2 / alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1, ite


def line_search(update, x0, g0, g, nstep=0, on=True):
    """
    `update` is the propsoed direction of update.

    Code adapted from scipy.
    """
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0) ** 2]
    torch.norm(x0) / torch.norm(update)

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]  # If the step size is so small... just return something
        x_est = x0 + s * update
        g0_new = g(x_est)
        phi_new = _safe_norm(g0_new) ** 2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new

    if on:
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=1e-2)
    if (not on) or s is None:
        s = 1.0
        ite = 0

    x_est = x0 + s * update
    if s == tmp_s[0]:
        g0_new = tmp_g0[0]
    else:
        g0_new = g(x_est)
    return x_est, g0_new, x_est - x0, g0_new - g0, ite


def rmatvec(part_Us, part_VTs, x):
    # Compute x^T(-I + UV^T)
    # x: (N, D)
    # part_Us: (N, D, L_thres)
    # part_VTs: (N, L_thres, D)
    if part_Us.nelement() == 0:
        return -x
    xTU = torch.einsum("bd, bdl -> bl", x, part_Us)  # (B, L_thres)
    return -x + torch.einsum("bl, bld -> bd", xTU, part_VTs)  # (B, D)


def matvec(part_Us, part_VTs, x):
    # Compute (-I + UV^T)x
    # x: (B, D)
    # part_Us: (B, D, L_thres)
    # part_VTs: (B, L_thres, D)
    if part_Us.nelement() == 0:
        return -x
    VTx = torch.einsum("bld, bd -> bl", part_VTs, x)  # (B, L_thres)
    return -x + torch.einsum("bdl, bl -> bd", part_Us, VTx)  # (B, D)
