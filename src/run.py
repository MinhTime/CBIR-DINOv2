"""End-to-end pipeline: download -> fine-tune -> extract -> Lw -> evaluate (-> alpha-QE).

Examples:
    python -m src.run --version b3                 # train + evaluate B3
    python -m src.run --version b4                 # B3 model + alpha-QE rows
    python -m src.run --version b3 --no-train      # evaluate the frozen backbone only
    python -m src.run --version b3 --epochs 1 --data-root /data/cbir
"""
import argparse
import json
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch

from .config import PRESETS, get_config
from .data import cid2filename, disk, download_test, download_train, load_gnd
from .evaluate import eval_ranks
from .extract import extract, ms_power, variants
from .model import HEADS, WEIGHTS, build_model, load_weights
from .qe import aqe
from .train import train
from .whiten import whitenapply, whitenlearn


class Pipeline:
    def __init__(self, cfg, device):
        self.cfg, self.device, self.results = cfg, device, []

    def _ext(self, net, paths, scales, **kw):
        return extract(net, paths, self.cfg['test_imsize'], scales, heads=True, num_workers=self.cfg['num_workers'], **kw)

    def extract_testset(self, net, ds, scales, tag):
        g = load_gnd(self.cfg['data_root'], ds)
        d = f"{self.cfg['data_root']}/test/{ds}/jpg"
        db = self._ext(net, [f'{d}/{n}.jpg' for n in g['imlist']], scales, desc=f'{tag} {ds} db')
        q = self._ext(net, [f'{d}/{n}.jpg' for n in g['qimlist']], scales,
                      bbxs=[x['bbx'] for x in g['gnd']], desc=f'{tag} {ds} query')
        torch.save({'db': db, 'q': q}, f"{self.cfg['out_dir']}/vecs_{tag}_{ds}.pt")
        return db, q

    def learn_whitening(self, net, scales, tag):
        d = f"{self.cfg['data_root']}/train/retrieval-SfM-120k"
        db = pickle.load(open(f'{d}/retrieval-SfM-120k-whiten.pkl', 'rb'))
        qi, pi = np.array(db['qidxs']), np.array(db['pidxs'])
        sel = np.random.RandomState(0).permutation(len(qi))[:self.cfg['whiten_max_pairs']]
        qi, pi = qi[sel], pi[sel]
        used = np.unique(np.concatenate([qi, pi]))
        remap = {int(o): n for n, o in enumerate(used)}
        paths = [cid2filename(db['cids'][i], f'{d}/ims') for i in used]
        print(f'Lw: {len(qi)} pairs, {len(paths)} images')
        per_head = self._ext(net, paths, scales, desc=f'{tag} whiten')
        p = net.pool.p.item()
        q2, p2 = [remap[int(i)] for i in qi], [remap[int(i)] for i in pi]
        W = {}
        for head in HEADS:
            v = variants(per_head[head], ms_power(head, p))
            W[head] = {sc: whitenlearn(v[sc].numpy().astype(np.float64), q2, p2) for sc in v}
        pickle.dump(W, open(f"{self.cfg['out_dir']}/whiten_{tag}.pkl", 'wb'))
        return W

    def _add(self, tag, ds, head, sc, lw, p, r):
        row = dict(stage=tag, dataset=ds, head=head, scale=sc, Lw=lw, p=round(float(p), 3),
                   **{f'{k}_{m}': v for k, d in r.items() for m, v in d.items()})
        key = (tag, ds, head, sc, lw)
        self.results = [x for x in self.results if (x['stage'], x['dataset'], x['head'], x['scale'], x['Lw']) != key]
        self.results.append(row)
        print(f"{tag:16s} {ds:9s} {head:3s} {sc} Lw={int(lw)} | mAP E {r['E']['mAP']:5.2f}  "
              f"M {r['M']['mAP']:5.2f}  H {r['H']['mAP']:5.2f} | mP@10 M {r['M']['mP@10']:5.2f}  H {r['H']['mP@10']:5.2f}")

    def score(self, tag, vecs, p, whiten=None, heads=HEADS, qe=False):
        for ds, (dbh, qh) in vecs.items():
            gnd = load_gnd(self.cfg['data_root'], ds)['gnd']
            for head in heads:
                dbv, qv = variants(dbh[head], ms_power(head, p)), variants(qh[head], ms_power(head, p))
                for sc in ['ss', 'ms']:
                    for lw in ([True] if qe else ([False, True] if whiten else [False])):
                        X, Q = dbv[sc].numpy(), qv[sc].numpy()
                        if lw:
                            X, Q = whitenapply(X, *whiten[head][sc]), whitenapply(Q, *whiten[head][sc])
                        if qe:
                            Q = aqe(X, Q, n=self.cfg['qe_n'], alpha=self.cfg['qe_alpha'])
                        self._add(tag, ds, head, sc, lw, p, eval_ranks(np.argsort(-(X.T @ Q), axis=0), gnd))
        self.save()

    def save(self):
        json.dump(self.results, open(f"{self.cfg['out_dir']}/results.json", 'w'), indent=1)
        pd.DataFrame(self.results).to_csv(f"{self.cfg['out_dir']}/results.csv", index=False)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', default='b3', choices=sorted(PRESETS))
    ap.add_argument('--data-root')
    ap.add_argument('--out-dir')
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--lr', type=float)
    ap.add_argument('--finetune-blocks', type=int)
    ap.add_argument('--train-head', choices=['gem', 'cls'])
    ap.add_argument('--num-workers', type=int)
    ap.add_argument('--resume-path')
    ap.add_argument('--no-train', action='store_true', help='evaluate the frozen backbone (no fine-tuning)')
    ap.add_argument('--no-whiten', action='store_true', help='skip Lw (no rSfM-120k download if --no-train)')
    ap.add_argument('--pretrained', action='store_true',
                    help='evaluate trained weights from weights/<VERSION>/ (not in the repo, available on request)')
    ap.add_argument('--weights', help='path to a ckpt_last.pt to evaluate (no training)')
    ap.add_argument('--whiten', help='path to a saved Lw whiten .pkl (skips learning Lw)')
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = get_config(a.version, data_root=a.data_root, out_dir=a.out_dir, epochs=a.epochs, lr=a.lr,
                     finetune_blocks=a.finetune_blocks, train_head=a.train_head,
                     num_workers=a.num_workers, resume_path=a.resume_path)
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg['data_root'], exist_ok=True)
    os.makedirs(cfg['out_dir'], exist_ok=True)
    print('torch', torch.__version__, '| device', device, '|', disk(cfg['data_root']))
    print('config:', json.dumps(cfg, indent=1))

    if a.pretrained:
        a.weights, a.whiten = a.weights or WEIGHTS[a.version][0], a.whiten or WEIGHTS[a.version][1]
    t0 = time.time()
    download_test(cfg['data_root'])
    if (not a.no_train and not a.weights) or (not a.no_whiten and not a.whiten):
        download_train(cfg['data_root'])
    print(f'data ok ({(time.time() - t0) / 60:.1f} min)')

    tag = f"{a.version.upper()}-{'frozen' if a.no_train else 'finetune'}"
    if a.weights:
        net = load_weights(cfg, device, a.weights)
    elif a.no_train:
        net = build_model(cfg, device, finetune_blocks=0)
    else:
        net = train(cfg, device)[0]

    pipe = Pipeline(cfg, device)
    vecs = {ds: pipe.extract_testset(net, ds, cfg['multiscale'], tag) for ds in cfg['datasets']}
    p = net.pool.p.item()
    pipe.score(tag, vecs, p)
    if not a.no_whiten:
        if a.whiten:
            with open(a.whiten, 'rb') as f:
                W = pickle.load(f)
        else:
            W = pipe.learn_whitening(net, cfg['multiscale'], tag)
        pipe.score(tag, vecs, p, W)
        if cfg['qe']:
            pipe.score(f'{tag}+aQE', vecs, p, W, heads=['cls'], qe=True)
    print('Results saved to', cfg['out_dir'])


if __name__ == '__main__':
    main()
