"""Revisited Oxford/Paris evaluation (Easy / Medium / Hard), ported from cirtorch."""
import numpy as np


def compute_ap(ranks, nres):
    ap, step = 0., 1. / nres
    for j, r in enumerate(ranks):
        p0 = 1. if r == 0 else j / r
        p1 = (j + 1) / (r + 1)
        ap += (p0 + p1) * step / 2.
    return ap


def compute_map(ranks, gnd, kappas=(1, 5, 10)):
    nq = len(gnd)
    aps, prs, nempty = np.zeros(nq), np.zeros((nq, len(kappas))), 0
    for i in range(nq):
        qgnd = np.array(gnd[i]['ok'])
        if qgnd.shape[0] == 0:
            aps[i] = np.nan
            prs[i] = np.nan
            nempty += 1
            continue
        qgndj = np.array(gnd[i].get('junk', []))
        pos = np.arange(ranks.shape[0])[np.isin(ranks[:, i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.isin(ranks[:, i], qgndj)]
        k = ij = 0
        if len(junk):
            for ip in range(len(pos)):
                while ij < len(junk) and pos[ip] > junk[ij]:
                    k += 1
                    ij += 1
                pos[ip] -= k
        aps[i] = compute_ap(pos, len(qgnd))
        pos = pos + 1
        for j, kk in enumerate(kappas):
            kq = min(max(pos), kk)
            prs[i, j] = (pos <= kq).sum() / kq
    n = nq - nempty
    return np.nansum(aps) / n, np.nansum(prs, axis=0) / n


def eval_ranks(ranks, gnd):
    """ranks: Ndb x Nq. Returns {'E'|'M'|'H': {'mAP', 'mP@1', 'mP@5', 'mP@10'}} in %."""
    out = {}
    for name, ok, jk in [('E', ['easy'], ['junk', 'hard']),
                         ('M', ['easy', 'hard'], ['junk']),
                         ('H', ['hard'], ['junk', 'easy'])]:
        g = [{'ok': np.concatenate([q[k] for k in ok]), 'junk': np.concatenate([q[k] for k in jk])} for q in gnd]
        m, pr = compute_map(ranks, g)
        out[name] = dict(mAP=round(100 * m, 2), **{f'mP@{k}': round(100 * v, 2) for k, v in zip((1, 5, 10), pr)})
    return out
