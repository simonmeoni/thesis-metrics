# Dataset Catalog

This file tracks all datasets and their processing history.

## MIMIC-III Datasets

### MIMIC 4% Epsilon DPO-2

| Version | File Path | Processing | Evaluation Date | Linkage Attack | Report |
|---------|-----------|------------|-----------------|----------------|--------|
| **Original** | `data/mimic-iii/4%-dpo_2-yv5v4516.parquet` | None | 2025-11-20 | 85.94% | `md/2025-11-20_mimic_privacy_evaluation.md` |
| **Standard Rephrased** | `data/mimic-iii/4%-dpo_2-yv5v4516_rephrased_llama-3.3-70b-versatile.parquet` | Standard rephrasing with llama-3.3-70b | 2025-11-20 | 83.73% | `md/2025-11-20_mimic_rephrasing_analysis.md` |
| **Keyword Replaced** | `data/mimic-iii/4%-dpo_2-yv5v4516_keyword_replaced.parquet` | Keyword-aware rephrasing | 2025-11-21 | 28.91% | `md/2025-11-21_keyword_replacement_vs_rephrasing_comparison.md` |

**Metadata**:
- Documents: 2,636 total, 1,764 matched
- Epsilon: 4%
- Method: DPO iteration 2
- Source: `data/mimic-iii/privacy_dataset_filtered.parquet`

---

### MIMIC 6% Epsilon DPO-2

| Version | File Path | Processing | Evaluation Date | Linkage Attack | Report |
|---------|-----------|------------|-----------------|----------------|--------|
| **Original** | `data/mimic-iii/6%-dpo_2-8l6avevp.parquet` | None | 2025-11-20 | 89.72% | `md/2025-11-20_mimic_privacy_evaluation.md` |
| **Standard Rephrased** | `data/mimic-iii/6%-dpo_2-8l6avevp_rephrased_llama-3.3-70b-versatile.parquet` | Standard rephrasing with llama-3.3-70b | 2025-11-20 | 84.99% | `md/2025-11-20_mimic_rephrasing_analysis.md` |
| **Keyword Replaced** | `data/mimic-iii/6%-dpo_2-8l6avevp_keyword_replaced.parquet` | Keyword-aware rephrasing | 2025-11-21 | 30.68% | `md/2025-11-21_keyword_replacement_vs_rephrasing_comparison.md` |

**Metadata**:
- Documents: 2,765 total, 1,819 matched
- Epsilon: 6%
- Method: DPO iteration 2
- Source: `data/mimic-iii/privacy_dataset_filtered.parquet`

---

## AlpaCare Datasets

### AlpaCare DPO-1

| Version | File Path | Processing | Evaluation Date | Linkage Attack | Report |
|---------|-----------|------------|-----------------|----------------|--------|
| **Original** | `data/alpacare/model=0fe1620_size=60000_step=dpo-1_sort=mes.parquet` | None | 2025-11-19 | 77.88% | `md/2025-11-20_alpacare_rephrasing_analysis.md` |
| **Standard Rephrased** | `data/alpacare/model=0fe1620_size=60000_step=dpo-1_sort=mes_rephrased_llama-3.3-70b-versatile.parquet` | Standard rephrasing with llama-3.3-70b | 2025-11-19 | 54.38% | `md/2025-11-20_alpacare_rephrasing_analysis.md` |
| **Keyword Replaced** | `data/alpacare/model=0fe1620_size=60000_step=dpo-1_sort=mes_keyword_replaced.parquet` | Keyword-aware rephrasing | 2025-11-21 | 14.41% | `md/2025-11-21_keyword_replacement_vs_rephrasing_comparison.md` |

**Metadata**:
- Documents: 3,823 total, 3,698 matched
- Method: DPO-1
- Source: `data/alpacare/alpacare_from_private_k-4000_full.parquet`

---

### AlpaCare DP-SFT

| Version | File Path | Processing | Evaluation Date | Linkage Attack | Report |
|---------|-----------|------------|-----------------|----------------|--------|
| **Original** | `data/alpacare/model=ovs3z1ey_size=60000_step=dp-sft_sort=mes.parquet` | None | 2025-11-19 | 24.78% | `md/2025-11-20_alpacare_rephrasing_analysis.md` |
| **Standard Rephrased** | `data/alpacare/model=ovs3z1ey_size=60000_step=dp-sft_sort=mes_rephrased_llama-3.3-70b-versatile.parquet` | Standard rephrasing with llama-3.3-70b | 2025-11-19 | 18.53% | `md/2025-11-20_alpacare_rephrasing_analysis.md` |
| **Keyword Replaced** | `data/alpacare/model=ovs3z1ey_size=60000_step=dp-sft_sort=mes_keyword_replaced.parquet` | Keyword-aware rephrasing | 2025-11-21 | 5.28% | `md/2025-11-21_keyword_replacement_vs_rephrasing_comparison.md` |

**Metadata**:
- Documents: 4,588 total, 6,062 matched
- Method: DP-SFT
- Source: `data/alpacare/alpacare_from_private_k-4000_full.parquet`

---

## Naming Convention

### Dataset Files
```
<dataset>/<base_name>_<processing>_<model>.parquet

Examples:
- Original: data/mimic-iii/4%-dpo_2-yv5v4516.parquet
- Standard: data/mimic-iii/4%-dpo_2-yv5v4516_rephrased_llama-3.3-70b-versatile.parquet
- Keyword:  data/mimic-iii/4%-dpo_2-yv5v4516_keyword_replaced.parquet
```

### Config Files
```
configs/dataset/<dataset>_<variant>.yaml

Examples:
- mimic_4pct_dpo2.yaml
- mimic_4pct_dpo2_rephrased.yaml
- mimic_4pct_dpo2_keyword_replaced.yaml
```

### Evaluation Outputs
```
outputs/YYYY-MM-DD/HH-MM-SS/
├── results.csv
└── config.yaml
```

### Reports
```
md/YYYY-MM-DD_<topic>.md

Examples:
- 2025-11-21_keyword_replacement_vs_rephrasing_comparison.md
- 2025-11-20_mimic_rephrasing_analysis.md
```

---

## Evaluation Results Summary

| Date | Dataset | Version | Linkage | Proximity | Output Dir |
|------|---------|---------|---------|-----------|------------|
| 2025-11-21 | MIMIC 4% DPO-2 | Keyword Replaced | 28.91% | 91.10% | `outputs/2025-11-21/10-28-21/` |
| 2025-11-21 | MIMIC 6% DPO-2 | Keyword Replaced | 30.68% | 94.61% | `outputs/2025-11-21/10-28-59/` |
| 2025-11-21 | AlpaCare DPO-1 | Keyword Replaced | 14.41% | 86.72% | `outputs/2025-11-21/10-29-01/` |
| 2025-11-21 | AlpaCare DP-SFT | Keyword Replaced | 5.28% | 57.21% | `outputs/2025-11-21/10-29-03/` |
| 2025-11-20 | MIMIC 4% DPO-2 | Standard Rephrased | 83.73% | 98.87% | - |
| 2025-11-20 | MIMIC 6% DPO-2 | Standard Rephrased | 84.99% | 99.73% | - |
| 2025-11-20 | MIMIC 4% DPO-2 | Original | 85.94% | 99.72% | - |
| 2025-11-20 | MIMIC 6% DPO-2 | Original | 89.72% | 99.95% | - |
| 2025-11-19 | AlpaCare DPO-1 | Standard Rephrased | 54.38% | 97.38% | - |
| 2025-11-19 | AlpaCare DP-SFT | Standard Rephrased | 18.53% | 66.66% | - |
| 2025-11-19 | AlpaCare DPO-1 | Original | 77.88% | 98.38% | - |
| 2025-11-19 | AlpaCare DP-SFT | Original | 24.78% | 69.65% | - |

---

## Quick Reference

### Best Privacy Protection (Linkage Attack Success Rate)
1. **AlpaCare DP-SFT + Keyword Replacement**: 5.28% ⭐
2. **AlpaCare DPO-1 + Keyword Replacement**: 14.41%
3. **AlpaCare DP-SFT + Standard Rephrasing**: 18.53%
4. **AlpaCare DP-SFT Original**: 24.78%
5. **MIMIC 4% DPO-2 + Keyword Replacement**: 28.91%
6. **MIMIC 6% DPO-2 + Keyword Replacement**: 30.68%

### Processing Pipeline Comparison
- **No Processing**: 24.78% - 89.72% attack success
- **+ Standard Rephrasing**: 18.53% - 84.99% attack success (2-24 pp improvement)
- **+ Keyword Replacement**: 5.28% - 30.68% attack success (13-55 pp additional improvement)

---

**Last Updated**: 2025-11-21
**Maintainer**: Update this file when creating new datasets or running evaluations
