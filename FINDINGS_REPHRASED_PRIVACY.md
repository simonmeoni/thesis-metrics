# Privacy Evaluation Findings: Impact of Rephrasing on Privacy Leakage

**Date**: November 19, 2025
**Evaluation**: AlpaCare DPO-1 and DP-SFT datasets (Original vs. Rephrased)

## Executive Summary

Rephrasing synthetic medical data with LLM (llama-3.3-70b-versatile) **significantly improves privacy protection** by reducing the success rate of privacy attacks. The most substantial improvement is observed in Linkage Attacks, where rephrasing reduces attack accuracy by up to 23.5 percentage points.

## Methodology

Four datasets were evaluated using standardized privacy attacks:
- AlpaCare DPO-1 (original and rephrased)
- AlpaCare DP-SFT (original and rephrased)

Privacy attacks tested:
- **Linkage Attack (TF-IDF)**: Attempts to match synthetic data to original private data
- **Proximity Attack (TF-IDF)**: Exploits cluster membership to infer privacy leakage
- Random baselines for comparison

## Detailed Results

### DPO-1 Dataset (3,698 documents)

| Attack Type | Original Accuracy | Rephrased Accuracy | Improvement | Attacks Prevented |
|------------|------------------|-------------------|-------------|------------------|
| **Linkage Attack** | 77.88% (2,880/3,698) | 54.38% (2,011/3,698) | **-23.50%** | 869 |
| **Proximity Attack** | 98.38% (3,638/3,698) | 97.38% (3,601/3,698) | **-1.00%** | 37 |
| Random Baseline | 0.11% | 0.11% | 0.00% | 0 |

### DP-SFT Dataset (6,062 documents)

| Attack Type | Original Accuracy | Rephrased Accuracy | Improvement | Attacks Prevented |
|------------|------------------|-------------------|-------------|------------------|
| **Linkage Attack** | 24.78% (1,502/6,062) | 18.53% (1,123/6,062) | **-6.25%** | 379 |
| **Proximity Attack** | 69.65% (4,222/6,062) | 66.66% (4,041/6,062) | **-2.99%** | 181 |
| Random Baseline | 0.05% | 0.05% | 0.00% | 0 |

## Key Insights

### 1. Linkage Attack Resistance
Rephrasing provides substantial protection against linkage attacks:
- **DPO-1**: 23.5 percentage point reduction (30.2% relative improvement)
- **DP-SFT**: 6.3 percentage point reduction (25.2% relative improvement)

This suggests that rephrasing effectively disrupts the TF-IDF similarity patterns that attackers exploit to match synthetic samples to their private counterparts.

### 2. Proximity Attack Resistance
Modest but consistent improvements in proximity attack resistance:
- **DPO-1**: 1.0 percentage point reduction
- **DP-SFT**: 3.0 percentage point reduction

Proximity attacks remain more successful because they exploit cluster membership information, which is preserved even after rephrasing.

### 3. Dataset-Specific Effects
- **DPO-1** (higher quality synthetic data) shows larger absolute improvements
- **DP-SFT** (more privacy-preserving) benefits from rephrasing despite already having lower baseline attack accuracy

### 4. Trade-off Analysis
Rephrasing introduces a privacy-utility trade-off:
- **Privacy gain**: Significant reduction in linkage attack success
- **Potential concern**: Proximity attacks remain highly effective on DPO-1 (97.38%)
- **Open question**: Impact on synthetic data utility and clinical validity (not measured in this evaluation)

## Implications for Thesis

1. **Rephrasing as a Privacy Enhancement Technique**
   Demonstrates that post-processing synthetic data with LLMs can provide an additional layer of privacy protection beyond the base synthetic data generation method.

2. **Differential Privacy Benefits**
   DP-SFT shows lower overall attack success rates, confirming that differential privacy during training provides fundamental privacy guarantees. Rephrasing offers complementary benefits.

3. **Multi-layered Defense**
   Best privacy protection comes from combining:
   - Differential privacy during training (DP-SFT)
   - Post-generation rephrasing (measured here)
   - Cluster-based anonymization (future work)

## Recommendations

1. **Adopt rephrasing for public release**: When sharing synthetic medical data, apply LLM-based rephrasing as a privacy enhancement step

2. **Investigate utility preservation**: Measure whether rephrased data maintains clinical utility (e.g., downstream classification tasks, medical validity)

3. **Explore advanced rephrasing strategies**: Test semantic-preserving paraphrasing vs. aggressive rephrasing to optimize privacy-utility trade-off

4. **Address proximity attacks**: Investigate additional defenses for cluster-based privacy leakage (e.g., cluster size thresholds, noise injection)

## Configuration Details

**Rephrasing Model**: llama-3.3-70b-versatile (Groq)
**Source Datasets**:
- `model=0fe1620_size=60000_step=dpo-1_sort=mes.parquet`
- `model=ovs3z1ey_size=60000_step=dp-sft_sort=mes.parquet`

**Rephrased Datasets**:
- `model=0fe1620_size=60000_step=dpo-1_sort=mes_rephrased_llama-3.3-70b-versatile.parquet`
- `model=ovs3z1ey_size=60000_step=dp-sft_sort=mes_rephrased_llama-3.3-70b-versatile.parquet`

**Attack Configuration**: TF-IDF vectorization, cosine similarity matching, cluster-aware proximity attacks

## Conclusion

**Rephrasing synthetic medical data with LLMs provides measurable privacy benefits**, particularly in defending against linkage attacks. The technique shows promise as a practical privacy enhancement layer, though further investigation is needed to ensure utility preservation and to address remaining vulnerabilities to proximity-based attacks.

---

*Generated from evaluation run: 2025-11-19 17:44:29*
*Configurations: `configs/dataset/alpacare_dpo_rephrased.yaml`, `configs/dataset/alpacare_dpsft_rephrased.yaml`*
