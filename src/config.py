"""Default configuration and per-version presets (B0..B4)."""
import math

DEFAULT = dict(
    data_root='/tmp/cbir_data',      # datasets are downloaded here
    out_dir='outputs/b3',            # results + checkpoints
    arch='dinov2_vitb14',
    datasets=['roxford5k', 'rparis6k'],
    test_imsize=518,                 # longest side, rounded down to a multiple of 14
    multiscale=[1, 1 / 2 ** 0.5, 1 / 2],
    gem_p=3.0,
    train_head='cls',                # descriptor used for loss + mining: 'gem' or 'cls'
    whiten_max_pairs=5000,           # q/p pairs used to learn Lw
    # ---- training ----
    train_imsize=518,
    finetune_blocks=2,               # number of last ViT blocks to train (ViT-B has 12)
    epochs=10,
    lr=1e-6,
    p_lr_mult=10,                    # lr of GeM p = lr * 10
    weight_decay=1e-6,
    lr_gamma=math.exp(-0.01),        # ExponentialLR
    margin=0.85,                     # contrastive loss margin
    query_size=2000,
    pool_size=22000,
    neg_num=5,
    batch_size=5,                    # tuples per optimizer step
    max_train_hours=8.0,
    eval_every_epoch=False,
    resume_path='',
    num_workers=4,
    amp=True,                        # fp16 autocast on CUDA
    seed=0,
    # ---- test-time ----
    qe=False,                        # alpha-QE rows (B4)
    qe_alpha=3,
    qe_n=2,
)

# Each version changes one factor vs the best previous one.
PRESETS = {
    'b0': dict(train_head='gem', finetune_blocks=2, lr=2e-6),
    'b1': dict(train_head='cls', finetune_blocks=2, lr=2e-6),
    'b2': dict(train_head='cls', finetune_blocks=4, lr=2e-6),
    'b3': dict(train_head='cls', finetune_blocks=2, lr=1e-6),
    'b4': dict(train_head='cls', finetune_blocks=2, lr=1e-6, qe=True),  # B3 model + alpha-QE
}


def get_config(version='b3', **overrides):
    cfg = dict(DEFAULT)
    cfg.update(PRESETS[version])
    cfg['out_dir'] = f'outputs/{version}'
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
