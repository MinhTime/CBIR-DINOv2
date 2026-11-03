"""Contrastive fine-tuning with hard-negative mining on rSfM-120k (as in the original GeM)."""
import json
import os
import time
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import Tuples, TupleSet
from .extract import extract
from .model import build_model


def contrastive_loss(x, margin, eps=1e-6):
    """x: D x (2+nneg) = [q, p, n1..nk] (cirtorch/layers/functional.py)."""
    d = (x[:, :1] - x[:, 1:] + eps).pow(2).sum(0).sqrt()
    lbl = torch.zeros_like(d)
    lbl[0] = 1
    return (0.5 * lbl * d.pow(2) + 0.5 * (1 - lbl) * (margin - d).clamp(min=0).pow(2)).sum()


def _collate(b):
    return b


def train(cfg, device, backbone=None, quick_eval=None):
    net = build_model(cfg, device, backbone=backbone)
    amp = cfg['amp'] and device.type == 'cuda'
    body = [p for n, p in net.named_parameters() if p.requires_grad and not n.startswith('pool.')]
    opt = torch.optim.Adam([{'params': body},
                            {'params': net.pool.parameters(), 'lr': cfg['lr'] * cfg['p_lr_mult'], 'weight_decay': 0}],
                           lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cfg['lr_gamma'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"trainable params: {n_train / 1e6:.1f}M (last {cfg['finetune_blocks']} / {len(net.blocks)} blocks + norm + p)")

    start, log = 0, []
    ck = cfg['resume_path'] or f"{cfg['out_dir']}/ckpt_last.pt"
    if os.path.exists(ck):
        s = torch.load(ck, map_location=device)
        net.load_state_dict(s['trainable'], strict=False)
        opt.load_state_dict(s['opt'])
        sched.load_state_dict(s['sched'])
        start, log = s['epoch'], s['log']
        print(f'resumed from {ck}, epoch {start}')

    ext = partial(extract, num_workers=cfg['num_workers'])
    T = Tuples(cfg, 'train')
    t_begin, ep_time = time.time(), 0.
    for ep in range(start, cfg['epochs']):
        if (time.time() - t_begin + ep_time) / 3600 > cfg['max_train_hours']:
            print('Early stop: about to exceed max_train_hours')
            break
        t0 = time.time()
        avg_nd = T.mine(net, ext)
        net.train()
        dl = DataLoader(TupleSet(T), batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=cfg['num_workers'], collate_fn=_collate, drop_last=True)
        tot, n = 0., 0
        for batch in tqdm(dl, desc=f'epoch {ep + 1}', mininterval=30):
            opt.zero_grad(set_to_none=True)
            for tup in batch:
                vecs = torch.cat([net(im.unsqueeze(0).to(device, non_blocking=True)) for im in tup]).t()
                loss = contrastive_loss(vecs, cfg['margin'])
                scaler.scale(loss).backward()
                tot += loss.item()
                n += 1
            scaler.step(opt)
            scaler.update()
        sched.step()
        ep_time = time.time() - t0
        rec = dict(epoch=ep + 1, loss=tot / max(1, n), avg_neg_dist=avg_nd, p=net.pool.p.item(), min=ep_time / 60)
        if cfg['eval_every_epoch'] and quick_eval is not None:
            rec['roxf_M'], rec['roxf_H'] = quick_eval(net)
        log.append(rec)
        print(rec)
        names = {k for k, p in net.named_parameters() if p.requires_grad}
        sd = {k: v for k, v in net.state_dict().items() if k in names}
        torch.save(dict(trainable=sd, opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep + 1, log=log, cfg=cfg),
                   f"{cfg['out_dir']}/ckpt_last.pt")
        json.dump(log, open(f"{cfg['out_dir']}/train_log.json", 'w'), indent=1)
    return net, log
