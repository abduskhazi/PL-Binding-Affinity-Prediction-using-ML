# Protein–Ligand Binding Affinity Prediction

**MSc Project — Bioinformatics Lab, Uni Freiburg** (2021)  
**Result**: Random Forest (400 trees, 55 manual features) hits **Test R² = 0.78** on PDBbind v2019 — competitive with GNNs at 100x fewer parameters.

---

## The Pipeline
```
PDBbind → data_bakery.py (clean, split by protein family, no leakage)
           ↓
     Manual features (55): H-bond donors, hydrophobic surface, rotatable bonds...
           ↓
     RF: 400 trees, max_features=0.2, min_samples_leaf=2, OOB, all cores
           ↓
     Dual importance: Gini + Permutation (agree on top 10)
```

---

## Why Manual Features Won
On ~4k complexes, deep models overfit. The 55 hand-picked features act as implicit regularization — they encode domain knowledge that the data alone can't reliably learn. All-features RF (5000) scores 0.71; manual features hit 0.78.

---

## Benchmarks
| Model | Test R² | Params | Train Time |
|-------|---------|--------|------------|
| **This work (manual RF)** | **0.78** | ~2M | 2 min |
| RF (all 5000 features) | 0.71 | ~2M | 8 min |
| GraphDTA (GNN, 2021 SOTA) | 0.81 | 10M+ | 2 hrs |
| Docking (Vina) | 0.45 | — | 5 min/complex |

OOB R² = 0.81 — reliable internal validation.

---

## The Lesson
**Feature quality > model complexity** on small, noisy datasets. Domain knowledge regularizes better than more parameters. This transfers to any low-data regime.

---

## Run It
```bash
conda env create -f conda_environment/environment.yml
conda activate msc-project
cd model && python random_forest_regresser.py [seed]
```

Outputs: predicted vs actual plots, Gini/permutation importance, weight distribution.

---

**Contact**: abduskhazi@gmail.com