# Guidelines for Claude: Dataset Management

This file contains instructions for maintaining the dataset catalog and tracking experimental work.

## Dataset Catalog Entry Requirements

When creating a new dataset or running a new evaluation, **ALWAYS update DATASET_CATALOG.md** with the following information:

### Required Fields for Each Dataset Entry

1. **Version Name**: Clear identifier (Original, Standard Rephrased, Keyword Replaced, etc.)

2. **File Path**: Complete path to the parquet file
   - Example: `data/mimic-iii/4%-dpo_2-yv5v4516_keyword_replaced.parquet`

3. **Processing Method**: Detailed description of what was done
   - Example: "Keyword-aware rephrasing with llama-3.3-70b-versatile"
   - Example: "Standard rephrasing after keyword replacement"
   - Include model name, temperature, any special settings

4. **Evaluation Date**: When the privacy evaluation was run (YYYY-MM-DD)

5. **Linkage Attack Score**: Percentage (from evaluation results)

6. **Proximity Attack Score**: Percentage (if applicable)

7. **Report Link**: Path to the markdown analysis report
   - Example: `md/2025-11-21_keyword_replacement_vs_rephrasing_comparison.md`

8. **Metadata**:
   - Number of documents (total and matched)
   - Privacy budget (epsilon) if applicable
   - Training method (DPO-1, DPO-2, DP-SFT, etc.)
   - Source dataset path

### Dataset Naming Convention

Follow this pattern for new processed datasets:

```
<base_name>_<processing_chain>.parquet

Examples:
- Original: 4%-dpo_2-yv5v4516.parquet
- Single processing: 4%-dpo_2-yv5v4516_keyword_replaced.parquet
- Chained processing: 4%-dpo_2-yv5v4516_keyword_replaced_rephrased.parquet
```

**Chain notation**:
- Use underscores to separate processing steps
- Order matters: read left-to-right for processing sequence
- `_keyword_replaced_rephrased` = keyword replacement THEN standard rephrasing

### Configuration File Requirements

When creating dataset configs in `configs/dataset/`:

1. **Config Name**: Match the dataset identifier
   - Example: `mimic_4pct_dpo2_keyword_replaced_rephrased.yaml`

2. **Required Fields**:
   ```yaml
   name: descriptive_dataset_name
   source_path: data/source/privacy_dataset.parquet
   synthetic_path: data/processed/dataset.parquet
   columns:
     private_instruction: instruction
     instruction: prompts
     response: response
     cluster_id: patient_id
     synthetic_text: column_to_evaluate
   ```

3. **Comments**: Add processing history as comments
   ```yaml
   # Processing history:
   # 1. Keyword replacement (2025-11-21)
   # 2. Standard rephrasing (2025-11-21)
   ```

## Workflow for New Datasets

### Step 1: Create the Dataset

Run rephrasing/processing command:
```bash
uv run rephrase rephrasing=METHOD \
  rephrasing.input_path="INPUT.parquet" \
  rephrasing.output_path="OUTPUT.parquet"
```

### Step 2: Create Config

Create `configs/dataset/DATASET_NAME.yaml` with appropriate settings.

### Step 3: Run Evaluation

```bash
uv run privacy-metrics dataset=DATASET_NAME
```

### Step 4: Update DATASET_CATALOG.md

Add new entry with all required fields (see above).

### Step 5: Create or Update Report

Create analysis report in `md/YYYY-MM-DD_TOPIC.md` or update existing report.

## Processing Chain Experiments

### Current Processing Methods

1. **None** (Original): Baseline synthetic data
2. **Standard Rephrasing**: Full-text LLM paraphrasing
3. **Keyword Replacement**: TF-IDF keyword extraction + synonym replacement

### Combination Experiments

Test different orderings and combinations:

| Combination | Naming | Purpose |
|------------|--------|---------|
| Keyword → Standard | `_keyword_replaced_rephrased` | Further obfuscation after keyword replacement |
| Standard → Keyword | `_rephrased_keyword_replaced` | Target keywords in already-paraphrased text |
| Keyword → Keyword | `_keyword_replaced_v2` | Multiple passes of keyword replacement |

**Documentation requirement**: For each combination, document:
- Hypothesis: Why this order/combination might improve privacy
- Expected outcome
- Actual results
- Comparison to single-method baselines

## Evaluation Results Tracking

### Updating DATASET_CATALOG.md After Evaluation

**CRITICAL**: After EVERY evaluation completes successfully, immediately update `DATASET_CATALOG.md`.

#### Step-by-Step Process

1. **Get Results**: Extract scores from `outputs/YYYY-MM-DD/HH-MM-SS/results.csv`
   - Linkage attack accuracy (%)
   - Proximity attack accuracy (%)

2. **Update Version Table**: In the appropriate dataset section, add or update the row:
   ```markdown
   | **Version Name** | `data/path/to/file.parquet` | Processing description | 2025-11-21 | XX.XX% | `md/report_link.md` |
   ```

3. **Add to Evaluation Results Summary**: Add row to the summary table at bottom:
   ```markdown
   | 2025-11-21 | Dataset Name | Version | XX.XX% | YY.YY% | `outputs/2025-11-21/HH-MM-SS/` |
   ```

4. **Update Quick Reference**: If this is a new best result, update the "Best Privacy Protection" section

#### Entry Format Examples

**Single processing:**
```markdown
| **Keyword Replaced** | `data/mimic-iii/4%-dpo_2-yv5v4516_keyword_replaced.parquet` | Keyword-aware rephrasing | 2025-11-21 | 28.91% | `md/2025-11-21_keyword_replacement.md` |
```

**Chained processing:**
```markdown
| **Keyword → Rephrased** | `data/mimic-iii/4%-dpo_2-yv5v4516_keyword_replaced_rephrased.parquet` | 1. Keyword replacement<br>2. Standard rephrasing | 2025-11-21 | XX.XX% | `md/2025-11-21_chained.md` |
```

### Minimum Information to Record

After each evaluation run:

1. **Attack Success Rates**:
   - Linkage Attack (TF-IDF)
   - Proximity Attack (TF-IDF)
   - Random baselines

2. **Dataset Statistics**:
   - Total documents
   - Matched documents (after keyword extraction)
   - Unique IDs

3. **Output Location**:
   - Path to `outputs/YYYY-MM-DD/HH-MM-SS/results.csv`

4. **Comparison**:
   - Improvement over previous method (percentage points)
   - Relative improvement (percentage)

## Report Writing Guidelines

When creating analysis reports in `md/`, follow the structure from existing reports to maintain consistency.

### Report Naming Convention

```
md/YYYY-MM-DD_<descriptive_topic>.md

Examples:
- 2025-11-21_keyword_replacement_vs_rephrasing_comparison.md
- 2025-11-21_chained_rephrasing_analysis.md
- 2025-11-20_mimic_rephrasing_analysis.md
```

### Required Report Sections

Every evaluation report MUST include these sections in order:

#### 1. **Header Block**
```markdown
# [Descriptive Title]

**Date**: YYYY-MM-DD
**Evaluation**: [Brief description of what was evaluated]
**Model**: [Model used for rephrasing, if applicable]
```

#### 2. **Executive Summary**
Must include:
- **Command used**: The exact command(s) that generated the results
  ```bash
  uv run privacy-metrics dataset=mimic_4pct_dpo2_keyword_replaced
  ```
- **Dataset(s) evaluated**: Name and file path
  - Example: "MIMIC 4% DPO-2 (`data/mimic-iii/4%-dpo_2-yv5v4516_keyword_replaced.parquet`)"
- **Key finding**: 1-2 sentence summary of the most important result
- **At-a-glance comparison table** (if comparing multiple methods/datasets)

#### 3. **Summary of Previous Experiments**
- What was done before this evaluation
- Previous baseline scores (linkage and proximity attacks)
- Reference to previous reports
- Context: Why this new evaluation was run

#### 4. **Results**
For each dataset evaluated, include:
- One subsection per dataset evaluated
- Include metadata: document counts, epsilon values, method
- Present results in tables with columns:
  - Attack type (Linkage, Proximity, Random Baseline)
  - Original/Baseline scores
  - New method scores
  - Improvement (absolute and relative)
  - Number of attacks prevented
- Always include context: "X attacks prevented out of Y total documents"

**Example table format:**
```markdown
### MIMIC 4% DPO-2 (1,764 documents)

| Attack Type | Original | New Method | Improvement | Attacks Prevented |
|------------|----------|------------|-------------|-------------------|
| Linkage Attack | 85.94% (1,516/1,764) | 28.91% (510/1,764) | -57.03 pp | 1,006 |
| Proximity Attack | 99.72% (1,759/1,764) | 91.10% (1,607/1,764) | -8.62 pp | 152 |
| Random Baseline | 0.23% (4/1,764) | 0.23% (4/1,764) | 0.00 pp | 0 |
```

#### 5. **Configuration Details**
- **Hydra config used**: Dataset config name (e.g., `dataset=mimic_4pct_dpo2_keyword_replaced`)
- Model and parameters
- Dataset file paths
- Column mappings
- Attack configuration
- Output directory path
- Evaluation completion time

#### 6. **Conclusion**
- Summary of key findings
- Main takeaways
- Next steps (if applicable)
- Links to related reports

### Writing Style Guidelines

1. **Use quantified comparisons**: "54.82 pp improvement" not "much better"
2. **Show both absolute and relative**: "28.91% (66.4% reduction from baseline)"
3. **Explain significance**: Don't just report numbers, interpret them
4. **Link to source data**: Reference previous reports and output directories
5. **Be specific**: "MIMIC 4% DPO-2" not "the MIMIC dataset"
6. **Bold key findings**: **Highlight important results** for scannability

## Quick Checklist for New Experiments

Before starting:
- [ ] Check DATASET_CATALOG.md for existing similar work
- [ ] Plan processing pipeline and name datasets accordingly
- [ ] Document hypothesis in notes

During processing:
- [ ] Use consistent naming conventions
- [ ] Save intermediate outputs
- [ ] Monitor job progress

After evaluation:
- [ ] Update DATASET_CATALOG.md with new entries
- [ ] Create or update analysis report
- [ ] Link report in catalog
- [ ] Document lessons learned

## Example: Complete New Dataset Entry

```markdown
### MIMIC 4% DPO-2 - Keyword Replaced Then Rephrased

| Version | File Path | Processing | Evaluation Date | Linkage Attack | Report |
|---------|-----------|------------|-----------------|----------------|--------|
| **Keyword → Standard** | `data/mimic-iii/4%-dpo_2-yv5v4516_keyword_replaced_rephrased.parquet` | 1. Keyword replacement (llama-3.3-70b)<br>2. Standard rephrasing (llama-3.3-70b) | 2025-11-21 | XX.XX% | `md/2025-11-21_chained_rephrasing.md` |

**Metadata**:
- Documents: 2,636 total, 1,764 matched
- Epsilon: 4%
- Method: DPO iteration 2
- Source: `data/mimic-iii/privacy_dataset_filtered.parquet`
- Processing chain: keyword_replaced → rephrased

**Hypothesis**: Applying standard rephrasing after keyword replacement may further reduce attack success by adding structural variation beyond keyword changes.

**Results**: [To be filled after evaluation]
```

## Error Prevention

### Common Mistakes to Avoid

1. **Forgetting to update catalog**: Always update after creating datasets
2. **Inconsistent naming**: Follow the naming convention exactly
3. **Missing source attribution**: Always link to reports and note processing history
4. **Incomplete metadata**: Fill in ALL required fields
5. **Not documenting hypothesis**: Write down WHY you're trying each experiment

### Validation Checklist

Before considering a dataset complete:
- [ ] Dataset file exists and is non-empty
- [ ] Config file created in configs/dataset/
- [ ] Evaluation completed successfully
- [ ] DATASET_CATALOG.md updated
- [ ] Report created or updated
- [ ] All scores and metadata recorded

---

**Last Updated**: 2025-11-21
**Purpose**: Maintain consistency and completeness in experimental tracking
