# Reliability Framework Documentation

This document covers the complete structure, mathematics, implementation, and certification procedures for the **Reliability Framework** within the Trustworthy AI Protein Engineering framework.

---

## 1. Scientific & Engineering Purpose

Traditional protein-engineering workflows typically prioritize **Predicted Performance** (e.g., predicted binding affinity, folding stability, or catalytic rate). However, models making these predictions often degrade or fail when evaluating candidates located in sparse regions of the sequence, evolutionary, or structural space.

This project introduces a dual-criteria prioritizing approach:
$$\text{Candidate Fitness} = f(\text{Predicted Performance}, \text{Prediction Reliability})$$

The **Reliability Framework** converts the scientific discoveries of the Sparsity Framework (namely, that Combined Sparsity is a causal predictor of thermodynamic antisymmetry error and epistasis error) into a practical engineering metric: **Reliability Score**.

Instead of asking only *What prediction was made?*, the framework asks: *How much confidence should be assigned to this prediction?*

---

## 2. Core Mathematics & Categories

The reliability of a prediction is modeled as the complement of its combined sparsity. A mutation with high combined sparsity lies in a sparse region of protein space, which correlates with high prediction errors, and thus exhibits low reliability.

### Reliability Score Formula

For each single-point mutation $i$:

$$\text{Reliability Score}_i = 1.0 - \text{Combined Sparsity}_i$$

where:
- $\text{Combined Sparsity}_i \in [0.0, 1.0]$ is the unified sparsity metric computed as the average of the available sparsity dimensions (Empirical, Evolutionary, and Structural).
- $\text{Reliability Score}_i \in [0.0, 1.0]$ represents the prediction confidence:
  - $\text{Reliability} = 1.0$ represents maximum reliability (complete confidence in model prediction).
  - $\text{Reliability} = 0.0$ represents minimum reliability (highly sparse region; extremely high prediction uncertainty).

### Engineering Reliability Categories

To facilitate decision-making, mutations are binned into four discrete engineering categories:

| Score Range | Reliability Category | Engineering Implication |
| :--- | :--- | :--- |
| $[0.75, 1.00]$ | **High Reliability** | Highly reliable prediction. Safe for high-throughput screening and direct ranking. |
| $[0.50, 0.75)$ | **Moderate Reliability** | Moderately reliable. Suitable for ranking with minor validation safety buffers. |
| $[0.25, 0.50)$ | **Low Reliability** | Low reliability. Use caution; high error risk. Requires experimental verification. |
| $[0.00, 0.25)$ | **Very Low Reliability** | Untrustworthy prediction. Highly prone to model failure. Should generally be avoided or heavily buffered. |

---

## 3. Directory Structure

The Reliability module is organized as follows:

```
src/reliability/
├── __init__.py               # Submodule exports
├── reliability_score.py      # Core calculation, pipeline engine & Athlete exercises
├── reliability_classifier.py # Categorization helper
└── reliability_validation.py # Scientific and schema validator (D402)
```

---

## 4. Dataset Specification (D402)

The final output is registered under Dataset ID **D402** and stored as a Snappy-compressed Parquet file.

### Schema Definition

| Column Name | Data Type | Nullable | Primary Key | Description |
| :--- | :--- | :---: | :---: | :--- |
| `mutation_id` | `string` | No | Yes | Unique identifier for the mutation |
| `combined_sparsity` | `double` | No | No | Calculated combined sparsity value in $[0.0, 1.0]$ |
| `reliability_score` | `double` | No | No | Derived reliability score in $[0.0, 1.0]$ |
| `reliability_category` | `string` | No | No | Assigned category: High, Moderate, Low, Very Low |

---

## 5. CLI Execution

### Generating Reliability Scores
To compute reliability scores and classify categories from the combined sparsity parquet:

```bash
PYTHONPATH=. python -m src.reliability.reliability_score \
  --sparsity data/intermediate/combined/combined_sparsity.parquet \
  --output data/final/reliability
```

### Auditing Output File
To independently run the scientific validation and schema checks on a generated parquet file:

```bash
PYTHONPATH=. python -m src.reliability.reliability_validation \
  --file data/final/reliability/reliability_scores.parquet
```

---

## 6. Validation Exercises

The module implements strict validation tests and exercises to ensure mathematical correctness, system-wide alignment, and sensitivity to errors.

### Athlete Exercises
Prior to processing, the engine verifies its scoring function against three hardcoded, scientifically explicit test cases:

1. **Athlete Exercise 1:**
   - Input: $\text{Combined Sparsity} = 0.10$
   - Expected Output: $\text{Reliability Score} = 0.90$
2. **Athlete Exercise 2:**
   - Input: $\text{Combined Sparsity} = 0.75$
   - Expected Output: $\text{Reliability Score} = 0.25$
3. **Athlete Exercise 3:**
   - Input: $\text{Combined Sparsity} = 1.00$
   - Expected Output: $\text{Reliability Score} = 0.00$

### Failure Injection Exercise
To verify that our validation suite is sensitive to calculation and bounds errors:
- An invalid value of $\text{Reliability} = 1.20$ is injected into the dataset (e.g., by executing `--inject-fault`).
- The validator MUST successfully identify this range constraint violation, block certification, and raise an appropriate validation error.
