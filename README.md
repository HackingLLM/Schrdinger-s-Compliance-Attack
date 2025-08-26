# Schrodinger's Compliance Attack

This is the repository for Schrödinger's Compliance Attack.

## Overview

We present a novel attack methodology that exploits the boundaries between safe and unsafe content in safety policies to extract targeted information.

### Key Insight

Our key finding is that models exhibit a "quantum superposition" state when presented with prompts containing both safe and unsafe content—when the volume of safe information exceeds unsafe information, models tend to provide sensitive information they shouldn't normally disclose.

### Attack Method

Our approach consists of the following steps:
1. Extract leaked safety policies from Chain-of-Thought (CoT) reasoning
2. Identify boundaries between allowed and disallowed behaviors  
3. Rephrase the original prompt based on these boundaries

## Features

Our generated attack prompts have the following key characteristics:

1. **High Success Rate**: Guaranteed to elicit responses under default sampling settings (may require multiple attempts)
2. **Information Completeness**: Provides sufficient information volume
3. **Quality Assurance**: [StrongReject](https://github.com/alexandrasouly/strongreject) score greater than 0.5, meeting standards for answering core requirements with adequate information

> **Note**: To ensure rephrasing quality, we need to call model APIs. The system will retrieve API keys locally, though you can substitute other APIs or locally deployed models.

## Quick Start

### Environment Installation

```bash
conda create -n s_attack python==3.11
```
Install all needed packages.

### Basic Attack

The current version integrates multiple attack methods and has obtained relatively comprehensive safety policies. You can directly use the following interface for attacks:

```bash
cd attack
python attack.py --prompt [your_prompt]
```

**Example Output**:
```
success [variant_prompt]
```

### Performance

We tested across multiple datasets with success rates between 40%-60%. Increasing retry attempts can improve attack success rates but will increase API token consumption:

```bash
cd attack
python attack.py --prompt [your_prompt] --max_trial N
```

## Complete Workflow

We support custom policy extraction and extension to other models. The complete attack workflow consists of 3 core components:

### 1. Extract Safety Policies

```bash
cd policy_extraction
python generate_policy.py
python generate_policy_gray.py
```

### 2. Iteratively Update Policies

```bash
python jailbreak_method.py
```

### 3. Generate Attack Prompts

```bash
python attack.py --prompt [your_prompt]
```

## Evaluation

Due to sampling uncertainty, we use [Jailbreak Oracle](https://github.com/jailbreak-oracle) to systematically evaluate our models and attack methods.

### Experimental Results

We conducted comprehensive evaluations with the following settings:
- **Sampling Parameters**: Temperature = 1.0, Top-p = 0.99
- **Success Criteria**: Combined original prompt + model-generated content with StrongReject score > 0.5

#### Attack Success Rate (ASR) Comparison

| Method | Success Rate | Description |
|--------|-------------|-------------|
| **Vanilla** | 3/90 (3.3%) | Original prompts without any modification |
| **Rephrase Only** | 18/90 (20.0%) | LLM-based rephrasing (with refusal mitigation strategies) |
| **Ours** | 40/90 (44.4%) | Our Schrödinger's Compliance Attack method |

> **Note**: For the "Rephrase Only" baseline, we mitigated and excluded cases where the LLM refused to rephrase prompts, ensuring fair comparison.

### Reproduction Results

You can clone locally and follow the README guide to reproduce our experimental data:

- **Customized Dataset** (containing 90 prompts, randomly sampled from AdvBench, JBB-Bench and AgentHarm, random_seed=2026): 44.4% success rate
- **HarmBench Dataset**: ~60% success rate

## Contributing

We welcome Issues and Pull Requests to improve this project.

## Disclaimer

This research is intended solely for academic research and AI safety evaluation purposes. Please do not use this tool for malicious purposes.