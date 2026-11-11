# B4 = B3 model + αQE

B4 does not train a new model. It reuses the B3 checkpoint and adds
α-weighted query expansion (αQE, α = 3, n = 2) at test time.
`results.csv` / `results.json` hold the `B3-finetune+aQE` rows (single- and multi-scale, with Lw).

| | ROxf M | ROxf H | RPar M | RPar H |
|---|:-:|:-:|:-:|:-:|
| B3 (no QE) | 72.0 | 52.8 | 89.8 | 77.1 |
| B4 (αQE 3/2) | 74.1 | 54.6 | 90.2 | 77.7 |

α = 3, n = 2 is the setting commonly used in the literature. The α/n grid tuned on the
rSfM-120k validation split is in [`../eval_v2/validation_qe/`](../eval_v2/validation_qe/)
(see the main README for why the validation choice, α = 0, n = 10, is not used here).

```bash
python -m src.run --version b4 --pretrained   # needs the B3 weights in weights/B3/ (available on request)
```
