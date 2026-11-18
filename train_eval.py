from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from utils import regression_metrics


def urbanroi_loss(y0, y1, roi_past, roi_future, alpha=0.1):
    # Eq. (17): last pre-entry ROI is repeated over all future steps.
    y0_target = roi_past[:, -1:].expand_as(y0)
    return F.mse_loss(y1, roi_future) + alpha * F.mse_loss(y0, y0_target)


def train_one_epoch(model, loader, optimizer, device, alpha=0.1, amp=False, scaler=None):
    model.train()
    total, n = 0.0, 0
    for g0, g1, xp, yf in loader:
        g0, g1 = g0.to(device), g1.to(device)
        xp, yf = xp.to(device), yf.to(device)
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast(device_type="cuda") if amp else nullcontext()
        with context:
            y0, y1, _ = model(g0, g1, xp)
            loss = urbanroi_loss(y0, y1, xp, yf, alpha)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item() * xp.size(0)
        n += xp.size(0)
    if n == 0:
        raise ValueError("Empty train loader")
    return total / n


@torch.no_grad()
def evaluate(model, loader, device, amp=False):
    model.eval()
    preds, gts = [], []
    for g0, g1, xp, yf in loader:
        g0, g1 = g0.to(device), g1.to(device)
        xp = xp.to(device)
        context = torch.autocast(device_type="cuda") if amp else nullcontext()
        with context:
            _, y1, _ = model(g0, g1, xp)
        preds.append(y1.float().cpu().numpy())
        gts.append(yf.numpy())
    if not preds:
        raise ValueError("Empty evaluation loader")
    return regression_metrics(np.concatenate(preds), np.concatenate(gts))
