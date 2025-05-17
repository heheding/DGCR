# Variational inference domain generalization with coupling relationships for industrial concept drift

This repository contains the official implementation of **Variational inference domain generalization with coupling relationships for industrial concept drift**, which is prepared for publication in Science China Information Sciences.

As the paper is still in the under review stage, I have removed the data processing. The full code will be filled in as soon as the paper is accepted.

## 🚀 Highlights

1. DGCR proposes a novel domain-label-free generalization framework integrating intra-domain variational coupling quantification and cross-domain invariant similarity. This architecture dynamically evaluates domain-specific contributions to eliminate pseudo-invariant representations caused by local optima.
2. Intra-domain variational coupling quantification removes the dependency on domain labels through theoretically derived conditional independent boundaries. The invariant similarity of cross domain pioneers the measurement of consistency in information theory across seen domains, jointly establishing rigorous mathematical foundations for domain generalization.
3. Comprehensive experiments on gas turbine performance and polyester esterification processes demonstrate DGCR's superiority, with quantitative comparisons and ablation studies validating enhanced generalization capabilities and feature disentanglement mechanisms.

## 🛠️ Installation

```bash
# Create a dedicated conda environment
conda create -n IDTL python=3.8.17
conda activate IDTL

# Install the package and dependencies
pip install -all .
```

## ▶️ Usage Example

Due to corporate confidentiality, we only provide examples on the public Gas turbine dataset.

```bash
# Run the DGCR pipeline
sh DGCR.sh
```

Adjust any configuration flags or data paths inside `DGCR.sh` and `DGCR.py` as needed for your setup.


## 📊 Performance Comparison

As shown in the table, IDTL outperforms the other methods in both MAE, RMSE and R²:
### Performance Comparison on CO and NOₓ Prediction Tasks

| Method | MAE (CO) | RMSE (CO) | R² (CO) | MAE (NOₓ) | RMSE (NOₓ) | R² (NOₓ) |
|--------|----------|-----------|--------|------------|-------------|---------|
| CIDA [1] | 0.8055 | 1.1027 | 0.3418 | 2.9546 | 3.7277 | 0.6503 |
| SAD [2] | _0.6410_ | _0.8488_ | _0.5224_ | 2.2573 | 2.7953 | 0.6945 |
| VDI [3] | 0.6678 | 0.8960 | 0.4651 | _2.1998_ | _2.7770_ | _0.7160_ |
| NU [4] | 0.6686 | 0.8862 | 0.3028 | 2.2970 | 2.8905 | 0.6487 |
| **DGCR (Ours)** | **0.5583** | **0.7416** | **0.6508** | **1.5544** | **1.9596** | **0.8635** |

> **Note**: Best results are **bolded**, second-best results are _italicized_.

## 📚 Citation

If you use this code in your research, please cite:

> DING H, HAO K R, CHEN L, et al. Variational inference domain generalization with coupling relationships for industrial concept drift. *Sci China Inf Sci*, for review.

## 🙏 Acknowledgements

This work was supported by:

* National Natural Science Foundation of China (62403121)
* Shanghai Pujiang Program (22PJ1423400)
* Shanghai Sailing Program (no. 22YF1401300)
  
## 📕 References
[1] Wang H, He H, Katabi D. Continuously indexed domain adaptation. Proc Int Conf Mach Learn, 2020: 9898–9907.

[2] Zhou Q, Gu Q, Pang J, et al. Self-adversarial disentangling for specific domain adaptation. IEEE Trans Pattern Anal Mach Intell, 2023, 45(7): 8954–8968.

[3] Xu Z, Hao G, He H, et al. Domain-indexing variational bayes: Interpretable domain index for domain adaptation. Proc Int Conf Learn Represent, 2023.

[4] Shi Z, Ming Y, Fan Y, et al. Domain generalization via nuclear norm regularization. Proc Mach Learn Res, 2024, 234: 179–201.
