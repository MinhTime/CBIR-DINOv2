"""DINOv2 backbone + GeM pooling; returns GeM, CLS and average-pooled descriptors."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PATCH

HEADS = ['gem', 'cls', 'avg']                       # heads used by the training runs (B0-B4)
ABLATION_HEADS = ['gem_shift', 'gem_signed']        # GeM variants without ReLU (paper v2, Sec. ablation)
ALL_HEADS = HEADS + ABLATION_HEADS


class GeM(nn.Module):
    """Generalized-mean pooling with learnable p (Radenović et al., TPAMI 2018)."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):  # B x D x h x w -> B x D
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), x.shape[-2:]).pow(1. / self.p).flatten(1)

    def signed(self, x):  # sign-preserving power mean: no ReLU, negative activations kept
        m = (x.sign() * x.abs().pow(self.p)).mean(dim=(2, 3))
        return m.sign() * m.abs().clamp(min=self.eps).pow(1. / self.p)


def flat_blocks(bb):
    out = []
    for b in bb.blocks:  # also supports BlockChunk
        out.extend([m for m in b if not isinstance(m, nn.Identity)] if isinstance(b, nn.ModuleList) else [b])
    return out


class RetrievalNet(nn.Module):
    """DINOv2 -> {gem: ReLU+GeM on patch tokens, cls: CLS token, avg: mean of patch tokens}, all L2-normalized.

    Only the last `finetune_blocks` blocks (+ final norm + GeM p) are trainable.
    """

    def __init__(self, backbone, p=3.0, finetune_blocks=0, train_head='cls', amp=True):
        super().__init__()
        self.backbone, self.pool = backbone, GeM(p)
        self.blocks = flat_blocks(backbone)
        self.n_frozen = len(self.blocks) - finetune_blocks
        self.train_head, self.amp = train_head, amp
        for prm in backbone.parameters():
            prm.requires_grad = False
        for b in self.blocks[self.n_frozen:]:
            for prm in b.parameters():
                prm.requires_grad = True
        if finetune_blocks > 0:
            for prm in backbone.norm.parameters():
                prm.requires_grad = True
        self.dim = backbone.embed_dim

    def forward(self, x, heads=False):
        bb = self.backbone
        B, _, H, W = x.shape
        h, w = H // PATCH, W // PATCH
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp and x.is_cuda):
            with torch.no_grad():                               # frozen part: no graph stored
                t = bb.prepare_tokens_with_masks(x)
                for blk in self.blocks[:self.n_frozen]:
                    t = blk(t)
            for blk in self.blocks[self.n_frozen:]:
                t = blk(t)
            t = bb.norm(t)
        nreg = getattr(bb, 'num_register_tokens', 0)
        pt = t[:, 1 + nreg:].float()                           # B x N x D patch tokens
        fmap = pt.transpose(1, 2).reshape(B, -1, h, w)
        o = {'gem': F.normalize(self.pool(F.relu(fmap)), dim=-1),   # ReLU: ViT tokens can be negative
             'cls': F.normalize(t[:, 0].float(), dim=-1),
             'avg': F.normalize(pt.mean(1), dim=-1),
             # ablations (eval only): GeM after shifting each channel to be non-negative (x - min), and signed GeM
             'gem_shift': F.normalize(self.pool(fmap - fmap.amin(dim=(2, 3), keepdim=True)), dim=-1),
             'gem_signed': F.normalize(self.pool.signed(fmap), dim=-1)}
        return o if heads else o[self.train_head]


def load_backbone(arch='dinov2_vitb14', device='cpu'):
    return torch.hub.load('facebookresearch/dinov2', arch).to(device).eval()


def build_model(cfg, device, finetune_blocks=None, backbone=None):
    bb = backbone if backbone is not None else load_backbone(cfg['arch'], device)
    fb = cfg['finetune_blocks'] if finetune_blocks is None else finetune_blocks
    return RetrievalNet(bb, p=cfg['gem_p'], finetune_blocks=fb,
                        train_head=cfg['train_head'], amp=cfg['amp']).to(device)


# Default location of trained weights (not in the repository; available on request).
# B4 = B3 weights + alpha-QE at test time.
WEIGHTS = {v: (f'weights/{v.upper()}/ckpt_last.pt', f'weights/{v.upper()}/whiten.pkl')
           for v in ['b0', 'b1', 'b2', 'b3']}
WEIGHTS['b4'] = WEIGHTS['b3']


def load_weights(cfg, device, path):
    """Build the model and load a fine-tuned checkpoint (only the trained blocks + norm + GeM p are stored)."""
    net = build_model(cfg, device)
    s = torch.load(path, map_location=device)
    r = net.load_state_dict(s.get('trainable', s), strict=False)
    if r.unexpected_keys:
        raise ValueError(f'{path} does not match finetune_blocks={cfg["finetune_blocks"]}: {r.unexpected_keys[:3]}')
    print(f'loaded weights from {path} (epoch {s.get("epoch", "?")})')
    return net.eval()
