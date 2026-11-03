"""Dataset download (ROxford5k / RParis6k / rSfM-120k), image loading and training tuples."""
import glob
import os
import pickle
import shutil
import subprocess

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet normalization (DINOv2)
PATCH = 14
CMP = 'http://cmp.felk.cvut.cz/cnnimageretrieval/data'


# ---------------------------------------------------------------- download
def sh(cmd):
    print('$', cmd)
    subprocess.run(cmd, shell=True, check=True)


def disk(path):
    u = shutil.disk_usage(path)
    return f'{path}: free {u.free / 2**30:.1f} GB / total {u.total / 2**30:.1f} GB'


def remote_size_gb(url):
    try:
        out = subprocess.run(f'curl -sIL --max-time 30 "{url}"', shell=True, capture_output=True, text=True).stdout
        sizes = [int(l.split(':')[1]) for l in out.splitlines() if l.lower().startswith('content-length')]
        return sizes[-1] / 2**30 if sizes else None
    except Exception:
        return None


def fetch(url, dst):
    if os.path.exists(dst):
        return
    try:
        sh(f'wget -q --tries=3 --timeout=60 -O "{dst}.part" "{url}" && mv "{dst}.part" "{dst}"')
    except subprocess.CalledProcessError:
        if os.path.exists(dst + '.part'):
            os.remove(dst + '.part')
        raise RuntimeError(f'Download failed: {url}')


def use_kaggle_input(root):
    """Use the Kaggle dataset rofiqsamanhudi/rparis6k-and-roxford5k when it is attached."""
    found = {}
    for ds in ['roxford5k', 'rparis6k']:
        g = sorted(glob.glob(f'/kaggle/input/**/{ds}/gnd_{ds}.pkl', recursive=True))
        if not (g and os.path.isdir(os.path.join(os.path.dirname(g[0]), 'jpg'))):
            continue
        src, d = os.path.dirname(g[0]), f'{root}/test/{ds}'
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(f'{d}/jpg'):
            # rparis6k is nested (jpg/paris/<landmark>/*.jpg) -> flat folder of symlinks
            os.makedirs(f'{d}/jpg')
            for fp in glob.glob(f'{src}/jpg/**/*.jpg', recursive=True):
                dstf = f'{d}/jpg/{os.path.basename(fp)}'
                if not os.path.exists(dstf):
                    os.symlink(fp, dstf)
        if not os.path.exists(f'{d}/gnd_{ds}.pkl'):
            shutil.copy(g[0], f'{d}/gnd_{ds}.pkl')
        gl = pickle.load(open(f'{d}/gnd_{ds}.pkl', 'rb'))
        miss = [n for n in gl['imlist'] + gl['qimlist'] if not os.path.exists(f'{d}/jpg/{n}.jpg')]
        if miss:
            print(f'{ds}: {len(miss)} images missing from Kaggle input, downloading from the web')
            shutil.rmtree(f'{d}/jpg')
            continue
        found[ds] = src
        print(f'{ds}: using Kaggle input {src} ({len(os.listdir(f"{d}/jpg"))} images)')
    return found


def download_test(root):
    """Same as cirtorch/utils/download.py (download_test)."""
    found = use_kaggle_input(root)
    if len(found) == 2:
        return
    src = {'oxford5k': ('https://www.robots.ox.ac.uk/~vgg/data/oxbuildings', ['oxbuild_images-v1.tgz']),
           'paris6k': ('https://www.robots.ox.ac.uk/~vgg/data/parisbuildings', ['paris_1-v1.tgz', 'paris_2-v1.tgz'])}
    for ds, (base, files) in src.items():
        if ('r' + ds) in found:
            continue
        jpg = f'{root}/test/{ds}/jpg'
        if os.path.isdir(jpg) and len(os.listdir(jpg)) > 1000:
            continue
        tmp = f'{root}/test/{ds}/tmp'
        os.makedirs(tmp, exist_ok=True)
        os.makedirs(jpg, exist_ok=True)
        for f in files:
            fetch(f'{base}/{f}', f'{tmp}/{f}')
            sh(f'tar -xzf "{tmp}/{f}" -C "{tmp}" && rm "{tmp}/{f}"')
        sh(f'find "{tmp}" -type f -name "*.jpg" -exec mv -t "{jpg}" {{}} +')
        shutil.rmtree(tmp)
    for ds, base in [('roxford5k', 'oxford5k'), ('rparis6k', 'paris6k')]:
        if ds in found:
            continue
        d = f'{root}/test/{ds}'
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(f'{d}/jpg'):
            os.symlink(f'{root}/test/{base}/jpg', f'{d}/jpg')
        fetch(f'{CMP}/test/{ds}/gnd_{ds}.pkl', f'{d}/gnd_{ds}.pkl')


def download_train(root):
    """Same as cirtorch download_train, but streams the tar to avoid using double disk space."""
    d = f'{root}/train/retrieval-SfM-120k'
    ims = f'{d}/ims'
    os.makedirs(ims, exist_ok=True)
    url = f'{CMP}/train/ims/ims.tar.gz'
    if not os.path.exists(f'{ims}/.done'):
        gb, free = remote_size_gb(url), shutil.disk_usage(root).free / 2**30
        print(f'ims.tar.gz ≈ {gb if gb is None else round(gb, 1)} GB, free {free:.1f} GB')
        if gb is not None and gb * 1.05 > free:
            raise RuntimeError('Not enough disk for rSfM-120k; change --data-root')
        sh(f'curl -L --retry 5 -sS "{url}" | tar -xzf - -C "{ims}" && touch "{ims}/.done"')
    for f in ['retrieval-SfM-120k.pkl', 'retrieval-SfM-120k-whiten.pkl']:
        fetch(f'{CMP}/train/dbs/{f}', f'{d}/{f}')
    print(disk(root))


def load_gnd(root, ds):
    with open(f'{root}/test/{ds}/gnd_{ds}.pkl', 'rb') as f:
        return pickle.load(f)


def cid2filename(cid, prefix):
    return os.path.join(prefix, cid[-2:], cid[-4:-2], cid[-6:-4], cid)


# ---------------------------------------------------------------- images
def load_img(path, imsize, bbx=None):
    img = Image.open(path).convert('RGB')
    if bbx is not None:
        img = img.crop(tuple(bbx))                       # test queries are cropped to their bbox
    img.thumbnail((imsize, imsize), Image.LANCZOS)        # downscale only, keep aspect ratio (as cirtorch)
    w, h = img.size
    w14, h14 = max(PATCH, w // PATCH * PATCH), max(PATCH, h // PATCH * PATCH)
    if (w14, h14) != (w, h):
        if w < PATCH or h < PATCH:
            img = img.resize((w14, h14), Image.BICUBIC)
        else:                                            # center-crop the < 14 px remainder
            l, t = (w - w14) // 2, (h - h14) // 2
            img = img.crop((l, t, l + w14, t + h14))
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.).permute(2, 0, 1)
    return (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]


class ImageList(Dataset):
    def __init__(self, paths, imsize, bbxs=None):
        self.paths, self.imsize, self.bbxs = paths, imsize, bbxs

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return load_img(self.paths[i], self.imsize, None if self.bbxs is None else self.bbxs[i])


# ---------------------------------------------------------------- training tuples
class Tuples:
    """Compact port of cirtorch TuplesDataset: (query, positive, hard negatives) from rSfM-120k."""

    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        d = f"{cfg['data_root']}/train/retrieval-SfM-120k"
        db = pickle.load(open(f'{d}/retrieval-SfM-120k.pkl', 'rb'))[split]
        self.images = [cid2filename(c, f'{d}/ims') for c in db['cids']]
        self.clusters, self.qpool, self.ppool = db['cluster'], db['qidxs'], db['pidxs']
        print(f'{split}: {len(self.images)} images, {len(self.qpool)} q/p pairs')

    def mine(self, net, extract_fn):
        """Pick random queries and mine hard negatives (one per cluster). Returns mean negative distance."""
        cfg = self.cfg
        sel = torch.randperm(len(self.qpool))[:cfg['query_size']]
        self.qidxs = [self.qpool[i] for i in sel]
        self.pidxs = [self.ppool[i] for i in sel]
        pool = torch.randperm(len(self.images))[:cfg['pool_size']].tolist()
        qv = extract_fn(net, [self.images[i] for i in self.qidxs], cfg['train_imsize'], desc='mine q')[1]
        pv = extract_fn(net, [self.images[i] for i in pool], cfg['train_imsize'], desc='mine pool')[1]
        ranks = torch.argsort(pv.t() @ qv, dim=0, descending=True)
        self.nidxs, dsum = [], 0.
        for q in range(len(self.qidxs)):
            clusters, nid, r = {self.clusters[self.qidxs[q]]}, [], 0
            while len(nid) < cfg['neg_num'] and r < len(pool):
                cand = pool[ranks[r, q]]
                if self.clusters[cand] not in clusters:
                    nid.append(cand)
                    clusters.add(self.clusters[cand])
                    dsum += (qv[:, q] - pv[:, ranks[r, q]]).norm().item()
                r += 1
            self.nidxs.append(nid)
        return dsum / max(1, sum(len(n) for n in self.nidxs))


class TupleSet(Dataset):
    def __init__(self, tuples):
        self.T = tuples

    def __len__(self):
        return len(self.T.qidxs)

    def __getitem__(self, i):
        idx = [self.T.qidxs[i], self.T.pidxs[i]] + self.T.nidxs[i]
        return [load_img(self.T.images[j], self.T.cfg['train_imsize']) for j in idx]
