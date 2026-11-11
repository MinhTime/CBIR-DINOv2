# CBIR: Landmark Retrieval with DINOv2 + GeM

A **Content-Based Image Retrieval (CBIR)** project for landmark (instance-level) retrieval.
Starting from **GeM** (Radenović et al., 2018), the CNN backbone is replaced with **DINOv2 ViT-B/14**,
then improved step by step. Each version changes **one factor** relative to the best previous version.

- **Training data:** rSfM-120k
- **Test data:** ROxford5k and RParis6k, mAP on the Medium and Hard protocols
- **Hardware:** Kaggle, T4 GPU

## Pipeline:

```
Image → DINOv2 ViT-B/14 → Global descriptor (GeM or CLS) → L2 norm → Lw whitening → Nearest-neighbor search → Query expansion (αQE)
```

## Results:

| Version | Change | ROxf Medium | ROxf Hard | RPar Medium | RPar Hard |
|---|---|:-:|:-:|:-:|:-:|
| Frozen | DINOv2 ViT-B/14, no fine-tuning, CLS + Lw (our protocol) | 70.1 | 49.8 | 88.6 | 76.0 |
| Frozen + αQE | Frozen + αQE (α = 3, n = 2) | 72.2 | 51.9 | 89.1 | 76.6 |
| B0 | Baseline: GeM, fine-tune last 2 blocks, lr 2e-6 (multi-scale) | 65.2 | 45.3 | 87.2 | 73.3 |
| B1 | Train on the CLS token instead of GeM | 72.1 | 52.9 | 89.7 | 76.5 |
| B2 | Fine-tune 4 blocks instead of 2 | 71.7 | 52.8 | 89.7 | 76.0 |
| B3 | Lower learning rate to 1e-6 | 72.0 | 52.8 | 89.8 | 77.1 |
| **B4** | **B3 + αQE (α = 3, n = 2) at test time** | **74.1** | **54.6** | **90.2** | **77.7** |

All numbers are mAP (%) with Lw whitening, single-scale unless noted. B2 and B3 both branch from B1.
The frozen rows use exactly the same images, resolution, whitening and evaluation code as B0-B4
(`results/eval_v2/frozen/`), so the comparison is fair: no QE vs no QE, QE vs QE.

![Results](figures/results.png)

### Model selection on validation

Versions and αQE settings are compared on the **rSfM-120k validation split**
(1,000 queries, 5,000 database images), not on ROxford/RParis (`results/eval_v2/validation/`).

| Model (CLS + Lw) | Frozen | B0 | B1 | B2 | B3 |
|---|:-:|:-:|:-:|:-:|:-:|
| val mAP | 90.96 | 91.13 | 93.13 | **93.56** | 92.73 |

Validation picks B2, but B1, B2 and B3 are within 0.8 val mAP and within 0.4 test mAP of each other,
so we treat them as **equivalent** (one seed each). The αQE grid on validation prefers α = 0, n = 10
(val mAP 94.71 for B2). This setting helps RParis (B2: 90.9 / 78.6) but hurts ROxford (69.3 / 49.2),
because the validation task has ~41 relevant images per query, far more than ROxford.
We therefore report the standard α = 3, n = 2 as the main setting and the validation choice alongside it.

### Extra checks (eval only, `results/eval_v2/`)

- **ReLU in GeM:** on B0, GeM without ReLU (signed power mean) gives ROxf M 64.9 vs 64.3 for standard GeM (+Lw),
  RPar 86.9 vs 85.9. The difference is small, so the GeM vs CLS gap is not explained by ReLU removing negative values.
  Shifting features to be positive (x - min) hurts badly (52.9 on ROxf).
- **Multi-scale and CLS (B6):** on B3, the default down-scaling `[1, 1/√2, 1/2]` hurts CLS (ROxf M 69.9 vs 72.0),
  while up-scaling `[√2, 1, 1/√2]` helps slightly (72.5; 75.1 with αQE). Small inputs, not multi-scale itself, hurt the CLS token.

**Findings**
- Fine-tuning on rSfM-120k beats frozen DINOv2 under the same protocol: +2.0 ROxf Medium / +3.0 Hard without QE, +1.9 / +2.8 with αQE.
- Training with the CLS token (B1) instead of GeM gives the largest gain: about +7 mAP on ROxford vs B0.
- B1, B2 and B3 are equivalent (~72 ROxf M); more training changes no longer help, the backbone is the limit.
- αQE adds about +2 mAP on ROxford (B4).

## Repository structure

```
src/          Python package with the full pipeline (same code as the notebooks, split into modules)
notebooks/    Kaggle notebooks for B0 to B3 (B4 is computed inside the B3 notebook) and eval_v2
results/      Results (csv, json) and logs for each version; results/eval_v2/ = evaluation-only checks
automation/   Script that pushes notebooks to Kaggle and collects results
figures/      Results chart
paper/        LaTeX source and PDF of the paper (Springer LNCS)
```

## Trained weights

The fine-tuned weights are **not stored in this repository**; they are available on request.
A checkpoint stores only the fine-tuned parameters (last 2 or 4 ViT blocks, final norm, GeM p);
the frozen layers are loaded from the official DINOv2 ViT-B/14 (torch hub).
To evaluate them, place the files as `weights/<VERSION>/ckpt_last.pt` and `weights/<VERSION>/whiten.pkl`:

```bash
python -m src.run --version b4 --pretrained         # B3 weights + αQE, no training, no rSfM-120k
python -m src.run --version b1 --weights weights/B1/ckpt_last.pt --whiten weights/B1/whiten.pkl
python -m src.eval_v2 --ckpt-root <dir with B0..B3 checkpoints>   # all eval_v2 checks
```

## How to run

### Option 1: Kaggle notebook

1. Upload a notebook from `notebooks/` to Kaggle.
2. Enable the T4 GPU and Internet, and add the dataset `rofiqsamanhudi/rparis6k-and-roxford5k`.
3. Click **Save & Run All**. Each run takes about 5 GPU hours.

### Option 2: Python package (`src/`)

Requires a CUDA GPU, Internet access, and about 45 GB of free disk for rSfM-120k.

```bash
pip install -r requirements.txt
python -m src.run --version b3                       # train + evaluate B3
python -m src.run --version b4                       # B3 model + αQE (best result)
python -m src.run --version b0 --data-root /data/cbir
python -m src.run --version b3 --no-train            # evaluate the frozen DINOv2 backbone
python -m src.run --version b3 --pretrained          # evaluate B3 weights from weights/B3/ (no training)
```

`--version` selects the preset (b0 to b4) defined in `src/config.py`. Results are written to `outputs/<version>/results.csv`.

| Module | Content |
|---|---|
| `config.py` | Default hyperparameters and per-version presets |
| `data.py` | Dataset download, image loading, training tuples with hard-negative mining |
| `model.py` | DINOv2 backbone + GeM / CLS / average descriptors |
| `extract.py` | Single- and multi-scale descriptor extraction |
| `train.py` | Contrastive loss and fine-tuning loop (with checkpoint resume) |
| `whiten.py` | Learned whitening (Lw) |
| `qe.py` | α-weighted query expansion |
| `evaluate.py` | Revisited Oxford/Paris mAP (Easy / Medium / Hard) |
| `run.py` | End-to-end pipeline and command-line interface |
| `eval_v2.py` | Evaluation-only checks: frozen baseline, validation selection, αQE grid, GeM ablation, B6 |
