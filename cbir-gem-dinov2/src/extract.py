"""Descriptor extraction (single- and multi-scale)."""
import gc

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import PATCH, ImageList
from .model import HEADS


def resize14(img, s):
    H, W = img.shape[-2:]
    nh, nw = max(PATCH, round(H * s / PATCH) * PATCH), max(PATCH, round(W * s / PATCH) * PATCH)
    return F.interpolate(img, size=(nh, nw), mode='bilinear', align_corners=False, antialias=True)


@torch.no_grad()
def extract(net, paths, imsize, scales=(1,), bbxs=None, desc='extract', heads=False, num_workers=4):
    """Return {scale: D x N} for the train head, or {head: {scale: D x N}} if heads=True."""
    net.eval()
    device = next(net.parameters()).device
    dl = DataLoader(ImageList(paths, imsize, bbxs), batch_size=1, shuffle=False, num_workers=num_workers)
    out = {h: {s: torch.zeros(net.dim, len(paths)) for s in scales} for h in HEADS}
    for i, img in enumerate(tqdm(dl, desc=desc, mininterval=10)):
        img = img.to(device, non_blocking=True)
        for s in scales:
            o = net(img if s == 1 else resize14(img, s), heads=True)
            for h in HEADS:
                out[h][s][:, i] = o[h].squeeze(0).cpu()
    del dl
    gc.collect()
    return out if heads else out[net.train_head]


def combine_ms(per_scale, p):
    """cirtorch extract_ms: l2n(mean_s(v_s^p)^(1/p)); p=1 -> plain mean (CLS/avg can be negative)."""
    v = torch.stack([per_scale[s] for s in per_scale])
    v = v.mean(0) if p == 1 else v.clamp(min=0).pow(p).mean(0).pow(1. / p)
    return F.normalize(v, dim=0)


def variants(per_scale, p):
    return {'ss': per_scale[1], 'ms': combine_ms(per_scale, p)}


def ms_power(head, p):
    return p if head == 'gem' else 1
