# eval_v2: evaluation-only experiments (no training)

Produced by `notebooks/eval_v2_kaggle.ipynb` (`python -m src.eval_v2`), one Kaggle T4 run (~2.2 h),
using the saved B0-B3 checkpoints. Full console output: `eval_v2.log`.

| Folder | What it contains |
|---|---|
| `validation/` | mAP, mP@10 and contrastive loss on the rSfM-120k **validation** split (1,000 queries, 5,000 database images, same-3D-model = relevant) for frozen DINOv2 and B0-B3, every head, with/without Lw |
| `selection/` | Model chosen on validation (B2, CLS + Lw) and the αQE setting chosen on validation |
| `validation_qe/` | αQE grid on validation: α ∈ {0,1,3,5,10}, n ∈ {1,2,3,5,10} for frozen, B2 and B3 |
| `frozen/` | Frozen DINOv2 ViT-B/14 (no fine-tuning) on ROxf/RPar under our protocol: GeM/CLS/Avg, single/multi-scale, ±Lw, ±αQE |
| `test_all/` | B0-B3 CLS descriptors on ROxf/RPar with no QE, αQE(3,2) and the validation-chosen αQE(0,10) |
| `ablation_gem/` | B0 checkpoint: standard GeM vs GeM without ReLU (`gem_signed`, signed power mean) vs shifted GeM (`gem_shift`, x - min) |
| `B6_upscale/` | B3 checkpoint with upscaling multi-scale `[√2, 1, 1/√2]` (`ms_up`) vs single-scale and the default `[1, 1/√2, 1/2]` (`ms`) |
