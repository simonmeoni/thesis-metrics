# Rephrasing Documentation

Text rephrasing module for data augmentation using Groq API.

## Quick Start

### Basic Usage

```bash
# Rephrase best-scoring generations
rephrase rephrasing=generation \
  rephrasing.input_path=data/input.parquet \
  rephrasing.output_path=data/output_rephrased.parquet

# Rephrase instruction column
rephrase rephrasing=instruction \
  rephrasing.input_path=data/input.parquet \
  rephrasing.output_path=data/output_rephrased.parquet
```

### API Key Setup

Set your Groq API key:
```bash
export GROQ_API_KEY='your-api-key-here'
```

## Configuration

### Changing Temperature

Override temperature from command line:
```bash
rephrase rephrasing=generation \
  rephrasing.input_path=data/input.parquet \
  rephrasing.output_path=data/output.parquet \
  rephrasing.temperature=0.9
```

**Temperature Guidelines:**
- `0.0-0.3`: More deterministic, conservative rephrasing
- `0.4-0.7`: Balanced rephrasing (default: 0.7)
- `0.8-1.0`: More creative, varied rephrasing

### Changing Model

```bash
rephrase rephrasing=generation \
  rephrasing.input_path=data/input.parquet \
  rephrasing.output_path=data/output.parquet \
  rephrasing.model=llama-3.1-8b-instant
```

**Available Models:**
- `llama-3.3-70b-versatile` (default, best quality)
- `llama-3.1-8b-instant` (faster, cheaper)
- `mixtral-8x7b-32768` (good for longer texts)

### Customizing Prompts

#### Method 1: Override from Command Line

```bash
rephrase rephrasing=generation \
  rephrasing.input_path=data/input.parquet \
  rephrasing.output_path=data/output.parquet \
  rephrasing.generation_prompt="Your custom prompt here. Original text: {text}"
```

**Important:** The prompt MUST include `{text}` placeholder where the original text will be inserted.

#### Method 2: Create Custom Config File

Create `configs/rephrasing/custom.yaml`:

```yaml
defaults:
  - default

mode: generation

# Custom prompt
generation_prompt: |
  Rephrase the following clinical text with maximum variation while preserving all medical facts.
  Use different sentence structures and synonyms where appropriate.

  Original: {text}

  Rephrased:

# Custom temperature for more variation
temperature: 0.9
```

Then use it:
```bash
rephrase rephrasing=custom \
  rephrasing.input_path=data/input.parquet \
  rephrasing.output_path=data/output.parquet
```

## Prompt Templates

### Default Generation Prompt

```
Please rephrase the following medical text while preserving all medical facts, dates, and clinical information.
Only provide the rephrased text without any additional commentary or explanations.

Original text:
{text}

Rephrased text:
```

### Default Instruction Prompt

```
You are a text rephrasing assistant. Your task is to rephrase the following medical instruction/question while:
1. Preserving the core medical question or intent
2. Changing the sentence structure and word choices to create a natural variation
3. Maintaining clarity and proper medical terminology when present
4. Keeping the same level of specificity

Original instruction:
{text}

Provide only the rephrased instruction without any additional commentary.
```

## Advanced Configuration

### Performance Tuning

```bash
rephrase rephrasing=generation \
  rephrasing.input_path=data/large.parquet \
  rephrasing.output_path=data/large_rephrased.parquet \
  rephrasing.max_concurrent=20 \
  rephrasing.batch_size=200
```

**Parameters:**
- `max_concurrent`: Number of concurrent API requests (default: 10)
- `batch_size`: Process in batches (default: 100)
- `max_tokens`: Maximum tokens per response (default: 2048)

### Different Columns

For instruction mode, specify which column to rephrase:

```bash
rephrase rephrasing=instruction \
  rephrasing.input_path=data/input.parquet \
  rephrasing.output_path=data/output.parquet \
  rephrasing.instruction_column=prompt
```

## Example Use Cases

### 1. Conservative Medical Rephrasing

Preserve medical terminology strictly:

```bash
rephrase rephrasing=generation \
  rephrasing.input_path=data/mimic.parquet \
  rephrasing.output_path=data/mimic_rephrased.parquet \
  rephrasing.temperature=0.3 \
  rephrasing.generation_prompt="Rephrase while keeping all medical codes, measurements, and terminology identical: {text}"
```

### 2. High-Variation Rephrasing

For privacy enhancement through linguistic diversity:

```bash
rephrase rephrasing=generation \
  rephrasing.input_path=data/alpacare.parquet \
  rephrasing.output_path=data/alpacare_rephrased.parquet \
  rephrasing.temperature=0.9 \
  rephrasing.generation_prompt="Completely rephrase this text using different words and sentence structures while preserving all facts: {text}"
```

### 3. Question Reformulation

For instruction/prompt datasets:

```bash
rephrase rephrasing=instruction \
  rephrasing.input_path=data/questions.parquet \
  rephrasing.output_path=data/questions_rephrased.parquet \
  rephrasing.temperature=0.8 \
  rephrasing.instruction_prompt="Reformulate this question differently while asking for the same information: {text}"
```

## Configuration Files

All configuration is in `configs/rephrasing/`:

- `default.yaml` - Base configuration with all options
- `generation.yaml` - For rephrasing generations (inherits from default)
- `instruction.yaml` - For rephrasing instructions (inherits from default)

### Full Configuration Options

```yaml
# Mode
mode: generation  # or "instruction"

# Paths
input_path: data/input.parquet
output_path: data/output_rephrased.parquet

# API Settings
model: llama-3.3-70b-versatile
temperature: 0.7
max_tokens: 2048

# Performance
max_concurrent: 10
batch_size: 100

# Instruction mode
instruction_column: instruction

# Prompts (with {text} placeholder)
instruction_prompt: |
  Your instruction rephrasing prompt here.
  Original: {text}
  Rephrased:

generation_prompt: |
  Your generation rephrasing prompt here.
  Original: {text}
  Rephrased:
```

## Output

### Generation Mode

Creates a new column `generation_rephrased` with rephrased text from the best-scoring generation.

### Instruction Mode

Replaces the specified column with rephrased text.

## Tips for Writing Custom Prompts

1. **Always include `{text}` placeholder** - This is where the original text will be inserted

2. **Be specific about what to preserve** - Medical facts, dates, measurements, etc.

3. **Request output format clearly** - "Only provide the rephrased text" prevents extra commentary

4. **Consider your use case**:
   - Privacy: Request maximum variation
   - Clinical utility: Request conservative changes
   - Data augmentation: Balance between variation and accuracy

5. **Test with small batches first** - Verify prompt behavior before large-scale rephrasing

## Troubleshooting

### API Rate Limits

Reduce `max_concurrent`:
```bash
rephrasing.max_concurrent=5
```

### Out of Memory

Reduce `batch_size`:
```bash
rephrasing.batch_size=50
```

### Inconsistent Output

Lower `temperature`:
```bash
rephrasing.temperature=0.3
```

### Need More Variation

Increase `temperature` and adjust prompt:
```bash
rephrasing.temperature=0.9 \
rephrasing.generation_prompt="Rephrase with maximum variation: {text}"
```

## Command Reference

View all configuration:
```bash
rephrase --help
rephrase --cfg job
```

Override multiple parameters:
```bash
rephrase rephrasing=generation \
  rephrasing.input_path=data/input.parquet \
  rephrasing.output_path=data/output.parquet \
  rephrasing.model=llama-3.1-8b-instant \
  rephrasing.temperature=0.8 \
  rephrasing.max_concurrent=20 \
  rephrasing.batch_size=200
```
