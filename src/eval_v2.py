"""Evaluation-only experiments (paper v2). No training.

  frozen        DINOv2 ViT-B/14 without fine-tuning, same protocol as B0-B4 (all heads, SS/MS, +-Lw, +-alpha-QE)
  validation    model selection on the rSfM-120k *val* split (frozen, B0-B3): retrieval mAP + contrastive loss,
                and the alpha-QE (alpha, n) grid, so that ROxford/RParis are used only for the final report
  ablation_gem  GeM without ReLU on frozen and B0: shifted GeM (x - min) and signed GeM
  B6_upscale    B3 CLS descriptor with the upscaled scale set [sqrt2, 1, 1/sqrt2]

Inputs: the outputs of the B0-B3 training kernels (ckpt_last.pt, whiten_<V>-finetune.pkl, vecs_<V>-finetune_<ds>.pt),
found anywhere below --ckpt-root (on Kaggle: /kaggle/input).

    python -m src.eval_v2 --ckpt-root /kaggle/input --out-dir outputs/eval_v2
"""
import argparse
import glob
import json
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch

from .config import get_config
from .data import cid2filename, disk, download_test, download_train, load_gnd
from .evaluate import compute_map, eval_ranks
from .extract import combine_ms, extract, ms_power, variants
from .model import ALL_HEADS, build_model, load_backbone
from .qe import aqe
from .whiten import whitenapply, whitenlearn

VERSIONS = ['B0', 'B1', 'B2', 'B3']
FINETUNE_BLOCKS = {'frozen': 0, 'B0': 2, 'B1': 2, 'B2': 4, 'B3': 2}
TRAINED_HEAD = {'frozen': 'cls', 'B0': 'gem', 'B1': 'cls', 'B2': 'cls', 'B3': 'cls'}
QE_ALPHAS = [0, 1, 3, 5, 10]
QE_NS = [1, 2, 3, 5, 10]
SQ2 = 2 ** 0.5
UP_SCALES = [SQ2, 1, 1 / SQ2]


class Collector:
    """Rows grouped by experiment; each group is written to <out_dir>/<group>/results.{csv,json}."""

    def __init__(self, out_dir):
        self.out_dir, self.rows = out_dir, {}

    def add(self, group, **row):
        self.rows.setdefault(group, []).append(row)
        print(group, {k: v for k, v in row.items() if k not in ('group',)})

    def add_test(self, group, stage, ds, head, scale, lw, qe, r):
        self.add(group, stage=stage, dataset=ds, head=head, scale=scale, Lw=lw, qe=qe,
                 **{f'{k}_{m}': v for k, d in r.items() for m, v in d.items()})

    def save(self):
        for g, rows in self.rows.items():
            d = f'{self.out_dir}/{g}'
            os.makedirs(d, exist_ok=True)
            json.dump(rows, open(f'{d}/results.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
            pd.DataFrame(rows).to_csv(f'{d}/results.csv', index=False)


def find_inputs(root):
    """Locate ckpt / whitening / saved test descriptors of B0-B3 below root."""
    found = {}
    for v in VERSIONS:
        w = sorted(glob.glob(f'{root}/**/whiten_{v}-finetune.pkl', recursive=True))
        if not w:
            raise FileNotFoundError(f'whiten_{v}-finetune.pkl not found below {root}')
        d = os.path.dirname(w[0])
        found[v] = dict(ckpt=f'{d}/ckpt_last.pt', whiten=w[0],
                        vecs={ds: f'{d}/vecs_{v}-finetune_{ds}.pt' for ds in ['roxford5k', 'rparis6k']})
        print(v, '->', d, sorted(os.listdir(d)))
    return found


def get_scale(per_scale, s):
    """Saved descriptor dicts use float scale keys; match them approximately."""
    for k in per_scale:
        if abs(float(k) - s) < 1e-6:
            return per_scale[k]
    raise KeyError(s)


def make_net(cfg, device, v, ckpt=None):
    net = build_model(cfg, device, finetune_blocks=FINETUNE_BLOCKS[v])
    if ckpt:
        s = torch.load(ckpt, map_location=device)
        r = net.load_state_dict(s.get('trainable', s), strict=False)
        assert not r.unexpected_keys, r.unexpected_keys[:3]
        print(f'{v}: loaded {ckpt} (epoch {s.get("epoch")}), GeM p = {net.pool.p.item():.3f}')
    return net.eval()


# ------------------------------------------------------------------ validation split
def build_val(cfg, nq, ndb, seed=0):
    """Fixed retrieval task on the rSfM-120k val split (3D models disjoint from train and from Oxford/Paris).

    Queries: nq unique val query images. Database: their positives + random val images (queries excluded),
    ndb images in total. Relevant = database images from the same 3D model (cluster) as the query.
    """
    d = f"{cfg['data_root']}/train/retrieval-SfM-120k"
    db = pickle.load(open(f'{d}/retrieval-SfM-120k.pkl', 'rb'))['val']
    rs = np.random.RandomState(seed)
    pairs, seen = [], set()
    for i in rs.permutation(len(db['qidxs'])):
        q = db['qidxs'][i]
        if q not in seen:
            seen.add(q)
            pairs.append((q, db['pidxs'][i]))
        if len(pairs) == nq:
            break
    qset = {q for q, _ in pairs}
    dbidx = list(dict.fromkeys(p for _, p in pairs if p not in qset))
    have = set(dbidx)
    for i in rs.permutation(len(db['cids'])):
        if len(dbidx) >= ndb:
            break
        if i not in qset and i not in have:
            dbidx.append(int(i))
            have.add(i)
    clusters = np.asarray(db['cluster'])
    qidx = [q for q, _ in pairs]
    pos_in_db = {p: j for j, p in enumerate(dbidx)}
    gnd = [{'ok': np.where(clusters[dbidx] == clusters[q])[0], 'junk': np.array([], dtype=int)} for q in qidx]
    paths = lambda ids: [cid2filename(db['cids'][i], f'{d}/ims') for i in ids]
    print(f'val task: {len(qidx)} queries, {len(dbidx)} database images, '
          f'{np.mean([len(g["ok"]) for g in gnd]):.1f} relevant / query')
    return dict(qpaths=paths(qidx), dbpaths=paths(dbidx), gnd=gnd,
                pos=[pos_in_db.get(p, -1) for _, p in pairs],
                qcl=clusters[qidx], dbcl=clusters[dbidx])


def mine_val_negatives(val, X, Q, n_neg=5):
    """Fixed hard negatives (one per other 3D model), mined once with the frozen model, reused for every model."""
    S = X.T @ Q
    negs = []
    for j in range(Q.shape[1]):
        used, nid = {val['qcl'][j]}, []
        for r in np.argsort(-S[:, j]):
            c = val['dbcl'][r]
            if c not in used:
                nid.append(int(r))
                used.add(c)
            if len(nid) == n_neg:
                break
        negs.append(nid)
    return negs


def val_loss(X, Q, val, negs, margin):
    """Mean contrastive loss on the fixed val tuples (L2 descriptors, as in training)."""
    tot, n = 0., 0
    for j, p in enumerate(val['pos']):
        if p < 0:
            continue
        dp = np.linalg.norm(Q[:, j] - X[:, p])
        dn = np.linalg.norm(Q[:, j:j + 1] - X[:, negs[j]], axis=0)
        tot += 0.5 * dp ** 2 + 0.5 * (np.clip(margin - dn, 0, None) ** 2).sum()
        n += 1
    return tot / max(n, 1)


def val_map(X, Q, gnd):
    m, pr = compute_map(np.argsort(-(X.T @ Q), axis=0), gnd)
    return round(100 * m, 2), round(100 * pr[2], 2)


# ------------------------------------------------------------------ main
class EvalV2:
    def __init__(self, cfg, device, inputs, out):
        self.cfg, self.device, self.inputs, self.out = cfg, device, inputs, out
        self.saved_w = {v: pickle.load(open(inputs[v]['whiten'], 'rb')) for v in VERSIONS}

    def ext(self, net, paths, scales, bbxs=None, desc=''):
        return extract(net, paths, self.cfg['test_imsize'], scales, bbxs=bbxs, desc=desc, heads=True,
                       num_workers=self.cfg['num_workers'], head_list=ALL_HEADS)

    def testset(self, net, ds, scales, tag):
        g = load_gnd(self.cfg['data_root'], ds)
        d = f"{self.cfg['data_root']}/test/{ds}/jpg"
        db = self.ext(net, [f'{d}/{n}.jpg' for n in g['imlist']], scales, desc=f'{tag} {ds} db')
        q = self.ext(net, [f'{d}/{n}.jpg' for n in g['qimlist']], scales, bbxs=[x['bbx'] for x in g['gnd']],
                     desc=f'{tag} {ds} query')
        return db, q

    def whitening(self, net, scales, tag, heads):
        d = f"{self.cfg['data_root']}/train/retrieval-SfM-120k"
        db = pickle.load(open(f'{d}/retrieval-SfM-120k-whiten.pkl', 'rb'))
        qi, pi = np.array(db['qidxs']), np.array(db['pidxs'])
        sel = np.random.RandomState(0).permutation(len(qi))[:self.cfg['whiten_max_pairs']]   # same pairs as B0-B3
        qi, pi = qi[sel], pi[sel]
        used = np.unique(np.concatenate([qi, pi]))
        remap = {int(o): n for n, o in enumerate(used)}
        per = self.ext(net, [cid2filename(db['cids'][i], f'{d}/ims') for i in used], scales, desc=f'{tag} whiten')
        q2, p2 = [remap[int(i)] for i in qi], [remap[int(i)] for i in pi]
        p = net.pool.p.item()
        W = {}
        for h in heads:
            v = {'ss': get_scale(per[h], 1)}
            if len(scales) > 1:
                v['ms'] = combine_ms(per[h], ms_power(h, p))
            W[h] = {sc: whitenlearn(x.numpy().astype(np.float64), q2, p2) for sc, x in v.items()}
        pickle.dump(W, open(f'{self.out.out_dir}/whiten_{tag}.pkl', 'wb'))
        return W

    def score(self, group, stage, vecs, p, W, heads, scales=('ss', 'ms'), qe=None):
        """vecs: {ds: (db, q)} with {head: {scale: D x N}}; qe: list of (alpha, n) applied on +Lw descriptors."""
        for ds, (dbh, qh) in vecs.items():
            gnd = load_gnd(self.cfg['data_root'], ds)['gnd']
            for h in heads:
                dv, qv = variants(dbh[h], ms_power(h, p)), variants(qh[h], ms_power(h, p))
                for sc in scales:
                    for lw in [False, True]:
                        X, Q = dv[sc].numpy(), qv[sc].numpy()
                        if lw:
                            X, Q = whitenapply(X, *W[h][sc]), whitenapply(Q, *W[h][sc])
                        self.out.add_test(group, stage, ds, h, sc, lw, '-', eval_ranks(np.argsort(-(X.T @ Q), axis=0), gnd))
                        if lw and qe and h == 'cls':
                            for a, n in qe:
                                r = eval_ranks(np.argsort(-(X.T @ aqe(X, Q, n=n, alpha=a)), axis=0), gnd)
                                self.out.add_test(group, stage, ds, h, sc, lw, f'a{a}_n{n}', r)
        self.out.save()

    # ---------------------------------------------------------------- val
    def val_descriptors(self, net, val, tag, heads):
        q = self.ext(net, val['qpaths'], [1], desc=f'{tag} val q')
        x = self.ext(net, val['dbpaths'], [1], desc=f'{tag} val db')
        return {h: (x[h][1].numpy(), q[h][1].numpy()) for h in heads}

    def val_rows(self, v, desc, val, negs, W):
        for h, (X, Q) in desc.items():
            loss = float(val_loss(X, Q, val, negs, self.cfg['margin']))
            for lw in [False, True]:
                Xw, Qw = (whitenapply(X, *W[h]['ss']), whitenapply(Q, *W[h]['ss'])) if lw else (X, Q)
                m, p10 = val_map(Xw, Qw, val['gnd'])
                self.out.add('validation', model=v, head=h, trained_head=h == TRAINED_HEAD[v], scale='ss', Lw=lw,
                             val_mAP=m, val_mP10=p10, val_loss=round(loss, 4))
        self.out.save()

    def qe_grid(self, v, X, Q, val, W):
        X, Q = whitenapply(X, *W['cls']['ss']), whitenapply(Q, *W['cls']['ss'])
        m0, _ = val_map(X, Q, val['gnd'])
        self.out.add('validation_qe', model=v, alpha='-', n=0, val_mAP=m0)
        best = (m0, None)
        for a in QE_ALPHAS:
            for n in QE_NS:
                m, _ = val_map(X, aqe(X, Q, n=n, alpha=a), val['gnd'])
                self.out.add('validation_qe', model=v, alpha=a, n=n, val_mAP=m)
                if m > best[0]:
                    best = (m, (a, n))
        self.out.save()
        return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt-root', default='/kaggle/input')
    ap.add_argument('--data-root', default='/tmp/cbir_data')
    ap.add_argument('--out-dir', default='outputs/eval_v2')
    ap.add_argument('--val-queries', type=int, default=1000)
    ap.add_argument('--val-db', type=int, default=5000)
    ap.add_argument('--num-workers', type=int, default=4)
    a = ap.parse_args()

    cfg = get_config('b3', data_root=a.data_root, out_dir=a.out_dir, num_workers=a.num_workers)
    random.seed(0), np.random.seed(0), torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(cfg['data_root'], exist_ok=True)
    print('torch', torch.__version__, '| device', device, '|', disk(os.path.dirname(os.path.abspath(a.data_root)) or '/'))
    t0 = time.time()
    inputs = find_inputs(a.ckpt_root)
    download_test(cfg['data_root'])
    download_train(cfg['data_root'])
    print(f'data ok ({(time.time() - t0) / 60:.1f} min)')
    load_backbone(cfg['arch'], 'cpu')  # warm the torch.hub cache once

    out = Collector(a.out_dir)
    E = EvalV2(cfg, device, inputs, out)
    val = build_val(cfg, a.val_queries, a.val_db)
    ms = cfg['multiscale']
    std_qe = [(3, 2)]

    # 1) frozen: val + test (all heads, MS) + Lw ----------------------------------------------------------
    net = make_net(cfg, device, 'frozen')
    vd = E.val_descriptors(net, val, 'frozen', ALL_HEADS)
    negs = mine_val_negatives(val, *vd['cls'])           # fixed val tuples, reused for every model
    Wf = E.whitening(net, ms, 'frozen', ALL_HEADS)
    E.val_rows('frozen', vd, val, negs, Wf)
    vecs_f = {ds: E.testset(net, ds, ms, 'frozen') for ds in cfg['datasets']}
    pf = net.pool.p.item()
    val_desc = {'frozen': vd['cls']}
    del net
    torch.cuda.empty_cache()

    # 2) B0 checkpoint: val + GeM ablation on test -------------------------------------------------------
    net = make_net(cfg, device, 'B0', inputs['B0']['ckpt'])
    vd = E.val_descriptors(net, val, 'B0', ALL_HEADS)
    W0 = E.whitening(net, ms, 'B0', ALL_HEADS)
    E.val_rows('B0', vd, val, negs, W0)
    vecs_0 = {ds: E.testset(net, ds, ms, 'B0') for ds in cfg['datasets']}
    p0 = net.pool.p.item()
    E.score('ablation_gem', 'B0', vecs_0, p0, W0, ['gem', 'gem_shift', 'gem_signed', 'avg', 'cls'])
    del net, vecs_0
    torch.cuda.empty_cache()
    E.score('ablation_gem', 'frozen', vecs_f, pf, Wf, ['gem', 'gem_shift', 'gem_signed', 'avg', 'cls'])

    # 3) B1-B3: val with their own (saved) whitening -----------------------------------------------------
    for v in ['B1', 'B2', 'B3']:
        net = make_net(cfg, device, v, inputs[v]['ckpt'])
        vd = E.val_descriptors(net, val, v, ['gem', 'cls', 'avg'])
        E.val_rows(v, vd, val, negs, E.saved_w[v])
        val_desc[v] = vd['cls']
        if v == 'B3':
            b3 = net
        else:
            del net
            torch.cuda.empty_cache()

    # 4) model selection on val (trained head, SS + Lw) and alpha-QE grid on val -------------------------
    rows = [r for r in out.rows['validation'] if r['trained_head'] and r['Lw']]
    sel = max(rows, key=lambda r: (r['val_mAP'], -r['val_loss']))
    print('selected on val:', sel)
    out.add('selection', **sel)
    best = {}
    for v in dict.fromkeys(['frozen', sel['model'], 'B3']):     # B3 = the model reported as B4 in v1
        if v in val_desc:
            best[v] = E.qe_grid(v, *val_desc[v], val, Wf if v == 'frozen' else E.saved_w[v])
    ref = sel['model'] if sel['model'] in best else 'B3'
    chosen = best[ref][1]                                      # None = no QE is best on val
    out.add('selection', model=f'alpha-QE (grid on {ref})', chosen_alpha=chosen[0] if chosen else '-',
            chosen_n=chosen[1] if chosen else 0, val_mAP=best[ref][0])
    out.save()
    qe_list = list(dict.fromkeys(std_qe + ([chosen] if chosen else [])))

    # 5) test: frozen baseline (same protocol) and B0-B3 re-scored from their saved descriptors -----------
    E.score('frozen', 'frozen', vecs_f, pf, Wf, ['gem', 'cls', 'avg'], qe=qe_list)
    for v in VERSIONS:
        vecs = {}
        for ds, path in inputs[v]['vecs'].items():
            s = torch.load(path, map_location='cpu')
            vecs[ds] = (s['db'], s['q'])
        E.score('test_all', v, vecs, 3.0, E.saved_w[v], ['cls'], qe=qe_list)  # p only matters for MS GeM

    # 6) B6: B3 with upscaled scale set [sqrt2, 1, 1/sqrt2] ------------------------------------------------
    W_up = E.whitening(b3, UP_SCALES, 'B3_up', ['cls'])
    for ds, path in inputs['B3']['vecs'].items():
        s = torch.load(path, map_location='cpu')
        up_db, up_q = E.testset(b3, ds, [SQ2], 'B3_up')
        gnd = load_gnd(cfg['data_root'], ds)['gnd']
        for name, scl in [('ss', [1]), ('ms', [1, 1 / SQ2, 0.5]), ('ms_up', UP_SCALES)]:
            def pick(src, up):
                return {k: (get_scale(up['cls'], k) if abs(k - SQ2) < 1e-6 else get_scale(src['cls'], k)) for k in scl}
            dbv, qv = pick(s['db'], up_db), pick(s['q'], up_q)
            X = combine_ms(dbv, 1).numpy() if len(scl) > 1 else dbv[1].numpy()
            Q = combine_ms(qv, 1).numpy() if len(scl) > 1 else qv[1].numpy()
            Wsc = W_up['cls']['ms'] if name == 'ms_up' else E.saved_w['B3']['cls'][name]
            for lw in [False, True]:
                Xw, Qw = (whitenapply(X, *Wsc), whitenapply(Q, *Wsc)) if lw else (X, Q)
                out.add_test('B6_upscale', 'B3', ds, 'cls', name, lw, '-', eval_ranks(np.argsort(-(Xw.T @ Qw), axis=0), gnd))
                if lw:
                    for al, n in qe_list:
                        r = eval_ranks(np.argsort(-(Xw.T @ aqe(Xw, Qw, n=n, alpha=al)), axis=0), gnd)
                        out.add_test('B6_upscale', 'B3', ds, 'cls', name, lw, f'a{al}_n{n}', r)
        out.save()
    print(f'done in {(time.time() - t0) / 3600:.2f} h; results in {a.out_dir}')


if __name__ == '__main__':
    main()
