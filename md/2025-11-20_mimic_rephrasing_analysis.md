# Privacy Evaluation Findings: MIMIC-III Rephrasing Analysis

**Date**: November 20, 2025
**Evaluation**: MIMIC-III 4% and 6% epsilon DPO-2 datasets (Original vs. Rephrased)

## Executive Summary

Rephrasing MIMIC-III synthetic data with LLM (llama-3.3-70b-versatile) provides **modest but measurable privacy improvements**. Linkage attack accuracy is reduced by 2.2-4.7 percentage points, while proximity attacks see minimal improvement (0.2-0.9 percentage points). This represents a more modest improvement compared to AlpaCare datasets, where rephrasing reduced linkage attacks by 6-24 percentage points.

## Methodology

Four datasets were evaluated using standardized privacy attacks:
- MIMIC-III 4% epsilon DPO-2 (original and rephrased)
- MIMIC-III 6% epsilon DPO-2 (original and rephrased)

Privacy attacks tested:
- **Linkage Attack (TF-IDF)**: Attempts to match synthetic data to original private data
- **Proximity Attack (TF-IDF)**: Exploits cluster membership to infer privacy leakage
- Random baselines for comparison

## Detailed Results

### 4% Epsilon DPO-2 Dataset (1,764 documents)

| Attack Type | Original Accuracy | Rephrased Accuracy | Improvement | Attacks Prevented |
|------------|------------------|-------------------|-------------|------------------|
| **Linkage Attack** | 85.94% (1,516/1,764) | 83.73% (1,477/1,764) | **-2.21%** | 39 |
| **Proximity Attack** | 99.72% (1,759/1,764) | 98.87% (1,744/1,764) | **-0.85%** | 15 |
| Random Baseline | 0.23% (4/1,764) | 0.23% (4/1,764) | 0.00% | 0 |

### 6% Epsilon DPO-2 Dataset (1,819 documents)

| Attack Type | Original Accuracy | Rephrased Accuracy | Improvement | Attacks Prevented |
|------------|------------------|-------------------|-------------|------------------|
| **Linkage Attack** | 89.72% (1,632/1,819) | 84.99% (1,546/1,819) | **-4.73%** | 86 |
| **Proximity Attack** | 99.95% (1,818/1,819) | 99.73% (1,814/1,819) | **-0.22%** | 4 |
| Random Baseline | 0.11% (2/1,819) | 0.11% (2/1,819) | 0.00% | 0 |

## Key Insights

### 1. Modest Linkage Attack Reduction

Rephrasing provides measurable but modest protection against linkage attacks:
- **4% epsilon**: 2.21 percentage point reduction (2.6% relative improvement)
- **6% epsilon**: 4.73 percentage point reduction (5.3% relative improvement)

This is notably less effective than AlpaCare results (6-24% reduction), suggesting that MIMIC-III's clinical note structure or DPO-2 generation method makes rephrasing less impactful.

### 2. Minimal Proximity Attack Reduction

Proximity attacks remain highly effective even after rephrasing:
- **4% epsilon**: 0.85 percentage point reduction (98.87% still vulnerable)
- **6% epsilon**: 0.22 percentage point reduction (99.73% still vulnerable)

The near-perfect proximity attack success rates indicate that cluster membership information is largely preserved through rephrasing, confirming that text paraphrasing cannot address structural privacy leakage.

### 3. Epsilon Effect Remains Counter-Intuitive

Higher epsilon (6%) continues to show worse privacy than lower epsilon (4%):
- **Original**: 6% epsilon has 3.78% higher linkage attack rate than 4%
- **Rephrased**: 6% epsilon has 1.26% higher linkage attack rate than 4%

However, rephrasing is MORE effective on the 6% epsilon dataset (4.73% improvement vs 2.21%), suggesting that higher epsilon synthetic data may have more linguistic variation that benefits from rephrasing.

### 4. Dataset Comparison: MIMIC-III vs AlpaCare

**Linkage Attack Reduction Comparison:**

| Dataset | Original Accuracy | Rephrased Accuracy | Absolute Improvement | Relative Improvement |
|---------|------------------|-------------------|---------------------|---------------------|
| AlpaCare DPO-1 | 77.88% | 54.38% | **-23.50%** | 30.2% |
| AlpaCare DP-SFT | 24.78% | 18.53% | **-6.25%** | 25.2% |
| MIMIC-III 4% | 85.94% | 83.73% | **-2.21%** | 2.6% |
| MIMIC-III 6% | 89.72% | 84.99% | **-4.73%** | 5.3% |

**Key Differences:**
- AlpaCare benefits 5-10x more from rephrasing than MIMIC-III
- MIMIC-III has much higher baseline privacy leakage (86-90% vs 25-78%)
- Both datasets show minimal proximity attack improvement

## Possible Explanations

### Why Rephrasing is Less Effective on MIMIC-III

1. **Higher Baseline Privacy Leakage**:
   - MIMIC-III: 86-90% linkage attack success
   - AlpaCare DPO-1: 78% linkage attack success
   - AlpaCare DP-SFT: 25% linkage attack success
   - When baseline leakage is already high, there's less room for improvement

2. **Clinical Note Structure**:
   - MIMIC-III contains structured clinical notes with standardized medical terminology
   - Medical codes, measurements, and clinical phrases may survive rephrasing intact
   - AlpaCare may have more natural language variation that's more amenable to paraphrasing

3. **DPO-2 vs DPO-1/DP-SFT Generation Methods**:
   - DPO-2 may produce more semantically stable outputs
   - Different training objectives may affect how much linguistic diversity exists in the synthetic data

4. **Keyword Density**:
   - MIMIC-III: 65-66% common keywords between source and synthetic
   - High keyword overlap suggests core medical terms dominate the text
   - Rephrasing may only affect narrative portions, leaving medical identifiers unchanged

### Why 6% Epsilon Benefits More from Rephrasing

The 6% epsilon dataset shows nearly double the improvement (4.73% vs 2.21%):
- Higher epsilon may allow more linguistic variation in synthetic generation
- More varied synthetic text provides more opportunity for rephrasing to disrupt TF-IDF patterns
- Trade-off: Higher epsilon also has worse baseline privacy (89.72% vs 85.94%)

## Implications for Thesis

### 1. Rephrasing Provides Limited But Real Benefits for MIMIC-III

While less dramatic than AlpaCare results, rephrasing does reduce privacy leakage:
- **86 attacks prevented** on 6% epsilon dataset (1,632 → 1,546 linkage attacks)
- **39 attacks prevented** on 4% epsilon dataset (1,516 → 1,477 linkage attacks)

These improvements are meaningful but insufficient as a standalone privacy defense.

### 2. Baseline Privacy Leakage is the Primary Concern

MIMIC-III datasets exhibit extremely high privacy leakage even before rephrasing:
- **Linkage attacks**: 86-90% success rate
- **Proximity attacks**: >99% success rate

Improving generation-time privacy (e.g., lower epsilon, better DP mechanisms) is more critical than post-processing.

### 3. Proximity Attacks Remain the Dominant Threat

Across all datasets (AlpaCare and MIMIC-III):
- Proximity attacks consistently achieve >95% accuracy after rephrasing
- Text paraphrasing cannot address cluster membership leakage
- **Cluster-aware privacy defenses are essential**

### 4. Dataset-Specific Privacy Enhancement Strategies Needed

Privacy techniques cannot be universally applied:
- AlpaCare: Strong rephrasing benefits (6-24% reduction)
- MIMIC-III: Modest rephrasing benefits (2-5% reduction)
- Different datasets require tailored privacy strategies based on:
  - Data structure (clinical notes vs conversational text)
  - Generation method (DPO-1, DPO-2, DP-SFT)
  - Baseline privacy leakage levels

## Recommendations

### 1. Use Rephrasing as a Complementary Defense

For MIMIC-III-style clinical datasets:
- Apply rephrasing as part of a multi-layered privacy strategy
- Do not rely on rephrasing alone (only 2-5% improvement)
- Prioritize generation-time privacy improvements first

### 2. Focus on Cluster-Aware Privacy Mechanisms

Given >99% proximity attack success:
- Implement cluster size thresholds (k-anonymity)
- Test cluster noise injection techniques
- Evaluate synthetic cluster generation to break patient ID linkages
- Consider cluster-level differential privacy

### 3. Investigate DPO-2 Privacy Characteristics

Understanding DPO-2's unique behavior:
- Why does 6% epsilon have worse privacy than 4%?
- How does DPO-2 compare to DPO-1 and DP-SFT on MIMIC-III?
- Can generation-time modifications improve DPO-2 privacy?

### 4. Explore Targeted Medical Entity Processing

Since medical terminology may resist rephrasing:
- Analyze which specific medical terms drive linkage attacks
- Test medical entity generalization (e.g., specific drugs → drug class)
- Investigate medical code masking or perturbation
- Balance privacy improvement against clinical utility loss

### 5. Consider Privacy-Utility Trade-offs

While rephrasing improves privacy modestly:
- Measure impact on downstream clinical utility
- Assess whether 2-5% privacy gain justifies potential utility loss
- Compare cost-benefit vs alternative privacy techniques

## Comparison: MIMIC-III vs AlpaCare Rephrasing Effectiveness

### Absolute Privacy Improvement

| Metric | MIMIC-III 4% | MIMIC-III 6% | AlpaCare DPO-1 | AlpaCare DP-SFT |
|--------|--------------|--------------|----------------|-----------------|
| Linkage Reduction | **2.21%** | **4.73%** | **23.50%** | **6.25%** |
| Proximity Reduction | **0.85%** | **0.22%** | **1.00%** | **2.99%** |
| Total Attacks Prevented | 54 | 90 | 906 | 560 |

### Why the Difference?

1. **Text Characteristics**:
   - MIMIC-III: Structured clinical notes with standardized terminology
   - AlpaCare: More natural language variation in patient-doctor dialogues

2. **Baseline Privacy**:
   - MIMIC-III: 86-90% linkage attack success (high leakage ceiling)
   - AlpaCare DP-SFT: 25% linkage attack success (more room for improvement)
   - AlpaCare DPO-1: 78% linkage attack success (moderate leakage)

3. **Generation Methods**:
   - DPO-2 (MIMIC-III) may encode information more robustly
   - DPO-1 and DP-SFT (AlpaCare) may be more sensitive to linguistic variation

## Configuration Details

**Rephrasing Model**: llama-3.3-70b-versatile (Groq)

**Source Datasets**:
- `4%-dpo_2-yv5v4516.parquet` (4% epsilon, DPO-2)
- `6%-dpo_2-8l6avevp.parquet` (6% epsilon, DPO-2)

**Rephrased Datasets**:
- `4%-dpo_2-yv5v4516_rephrased_llama-3.3-70b-versatile.parquet`
- `6%-dpo_2-8l6avevp_rephrased_llama-3.3-70b-versatile.parquet`

**Column Mapping**:
- Private dataset: `instruction` column
- Synthetic dataset: `prompts` column
- Rephrased text: `generation_rephrased` column

**Attack Configuration**: TF-IDF vectorization, cosine similarity matching, patient ID-aware proximity attacks

## Conclusion

**Rephrasing MIMIC-III synthetic data provides modest but measurable privacy benefits**, reducing linkage attack accuracy by 2-5 percentage points. However, this improvement is substantially less than AlpaCare results (6-24% reduction), suggesting that:

1. **Dataset characteristics matter**: Clinical notes with standardized terminology are less amenable to rephrasing
2. **Baseline privacy is critical**: High initial leakage (86-90%) limits improvement potential
3. **Proximity attacks persist**: >99% success rate after rephrasing indicates structural privacy leakage
4. **Multi-layered defense needed**: Rephrasing should complement, not replace, generation-time privacy mechanisms

**Key Takeaway**: While rephrasing prevents 39-86 successful attacks per dataset (meaningful for individual privacy), the fundamental privacy challenges of MIMIC-III synthetic data require deeper interventions:
- Stronger differential privacy during generation
- Cluster-aware privacy mechanisms
- Medical entity-specific privacy techniques
- Careful evaluation of DPO-2 privacy characteristics

**Next Steps**:
1. Prioritize cluster-based privacy defenses (k-anonymity, noise injection)
2. Compare DPO-1/DP-SFT vs DPO-2 on MIMIC-III data
3. Investigate medical entity masking strategies
4. Measure rephrasing impact on clinical utility

---

*Generated from evaluation run: 2025-11-20 15:36:25*
*Configurations: `configs/dataset/mimic_4pct_dpo2.yaml`, `configs/dataset/mimic_6pct_dpo2.yaml`, `configs/dataset/mimic_4pct_dpo2_rephrased.yaml`, `configs/dataset/mimic_6pct_dpo2_rephrased.yaml`*
*Using corrected column mappings (private_instruction: instruction, instruction: prompts, synthetic_text: generation_rephrased)*
