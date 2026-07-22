# Publication Figures Module

This document outlines the design, implementation, and certification of the **Publication Figures Module** (Phase 16) within the Trustworthy AI Protein Engineering framework.

---

## 1. Scientific & Engineering Purpose

The primary objective of this module is to generate a comprehensive suite of publication-ready visual assets representing the entire computational discovery, validation, and engineering application pipeline of the project. These figures are optimized for both high-resolution digital publication (PNG format, 300 DPI) and vector-quality print dissemination (PDF format).

Every figure has been developed following strict styling rules (legible labels, consistent color mapping, no overlapping titles, and grid alignments) and has been certified through automated **Athlete Exercises** validating coordinate matches and mathematical categorization logic.

---

## 2. Directory Structure

The visual generation module is structured under `src/figures/` as follows:

```
src/figures/
├── __init__.py             # Package exports
├── generate_all.py         # Master runner coordinating all generation
├── discovery.py            # Figure F2 (Combined Sparsity vs. Antisymmetry Error)
├── validation.py           # Figure F3 (Combined Sparsity vs. Epistasis Error)
├── integrated.py           # Figure F4 (Coupling Error Analysis)
├── reliability.py          # Figure F5 (Reliability Score Distribution)
├── tkt_landscape.py        # Figures F6 & F7 (TKT Mutation Landscape plots)
├── candidate_discovery.py  # Figure F8 (Top Prioritized Candidates)
└── killer_figure.py        # Figure F9 (Flagship 8-Panel Causal Narrative Dashboard)
```

And corresponding unit/integration tests:

```
tests/figures/
└── test_figures.py         # Automated pytest suite validating execution & exports
```

---

## 3. Master Figure Index & Captions

The module produces nine distinct visual outputs, each stored in `figures/` as both `.png` and `.pdf` files:

### **Figure F1: Master Project Architecture**
- **Purpose**: Map out the three key phases of the computational ecosystem (Discovery, Validation, and prospective Application) to explain the data dependencies and model mechanics at a glance.
- **Caption**: *Figure F1: Master computational dataflow mapping out the Trustworthy AI Protein Engineering architecture. Source datasets from Megascale-D feed multi-dimensional sparsity models, which couple to thermodynamic inconsistency (Experiment 1) and epistasis prediction errors (Experiment 2). The validated model is utilized prospective-wise to evaluate the 11,400 single amino-acid substitutions across Transketolase (TKT), prioritizing candidate ranking with active-site distance controls.*

### **Figure F2: Sparsity vs. Thermodynamic Inconsistency (Discovery)**
- **Purpose**: Support the scientific discovery that combined mutation-space sparsity is a strong, statistically significant predictor of thermodynamic antisymmetry error.
- **Caption**: *Figure F2: Pairwise scatter mapping of Combined Sparsity versus Thermodynamic Antisymmetry Error (|Forward + Reverse| predictions). The red line indicates the linear regression fit. A shaded red band shows the 95% Confidence Interval. Pearson ($r$) and Spearman ($r_s$) correlation coefficients demonstrate a strong positive relationship ($r > 0.40$), certifying Combined Sparsity as a robust predictor of model prediction uncertainty.*

### **Figure F3: Sparsity vs. Epistasis Prediction Error (Validation)**
- **Purpose**: Validate that combined mutation-space sparsity predicts actual, real-world epistasis prediction failures against experimental deep mutational scanning measurements.
- **Caption**: *Figure F3: Double-mutant epistasis prediction error mapped against pair-level combined sparsity. Solid red line represents the regression fit surrounded by its 95% confidence interval. High combined sparsity correlates with significantly higher prediction errors ($r > 0.35$), confirming that the model's reliability degrades when evaluating candidates in sparse regions.*

### **Figure F4: Mechanistic Error Coupling (Integrated)**
- **Purpose**: Connect the scientific discovery (thermodynamic inconsistency) directly to the validation (experimental prediction failure), verifying the causal error coupling chain.
- **Caption**: *Figure F4: Mutation-level thermodynamic antisymmetry error plotted against double-mutant epistasis prediction error. The strong positive slope ($\beta$) and highly significant correlation ($p < 10^{-5}$) illustrate that thermodynamic inconsistency in single-mutation models propagates as prediction failure when combined into epistatic double-mutant designs.*

### **Figure F5: Prediction Reliability Score Distribution**
- **Purpose**: Categorize single-point mutations based on the mathematical formulation of prediction reliability: $\text{Reliability} = 1.0 - \text{Sparsity}$.
- **Caption**: *Figure F5: Distribution of engineering Reliability Scores across all single-point mutations. Vertical dashed lines mark the boundaries between High ($\ge 0.75$, green), Moderate ($[0.50, 0.75)$, yellow), Low ($[0.25, 0.50)$, orange), and Very Low ($< 0.25$, red) reliability categories. This score enables wet-lab screening prioritization based on prediction confidence.*

### **Figure F6: Transketolase (TKT) Reliability Landscape**
- **Purpose**: Map out prediction reliability across every single residue position in the 600-amino-acid enzyme.
- **Caption**: *Figure F6: Complete prospective mutational reliability landscape for Transketolase. Each of the 11,400 possible single substitutions is represented as a point colored by its engineering reliability category (Green: High, Yellow: Moderate, Red: Low/Very Low). This landscape maps the sequence-level boundaries where the predictive models can be safely trusted.*

### **Figure F7: TKT Mutational Industrial Fitness Distribution**
- **Purpose**: Visualize the distribution of engineering suitability across all 11,400 candidates under dual-criteria optimization.
- **Caption**: *Figure F7: Frequency distribution of TKT Industrial Fitness Scores. Suitability is modeled as a weighted sum of Predicted Stability ($50\%$), Prediction Reliability ($35\%$), and Evolutionary Plausibility ($15\%$), ensuring selected mutations are highly active yet safe for wet-lab screening.*

### **Figure F8: Top Prioritized Engineering Candidates**
- **Purpose**: Provide practical, prospective wet-lab recommendations.
- **Caption**: *Figure F8: Top 15 prospective Transketolase mutation candidates prioritized by Industrial Fitness Score. Bars are color-coded by prediction reliability class (Green: High, Yellow: Moderate). Annotations list raw fitness scores alongside predicted thermodynamic stability ($S$, kcal/mol) and reliability ($R$), delivering high-value, de-risked targets for experimental validation.*

### **Figure F9: Flagship Killer Figure**
- **Purpose**: Synthesize the entire scientific discovery, validation, and prospective engineering application narrative of the project in a single, high-impact multi-panel dashboard.
- **Caption**: *Figure F9: Flagship multi-panel visual dashboard illustrating the complete end-to-end scientific narrative of the Trustworthy AI Protein Engineering project across 8 panels. Panel A (1) displays the Combined Sparsity distribution foundation for Megascale-D. Panel B (2) shows the discovery that Combined Sparsity predicts thermodynamic antisymmetry error. Panel C (3) validates that Combined Sparsity predicts double-mutant epistasis prediction failure. Panel D (4) demonstrates the causal mechanistic coupling of single-mutant thermodynamic error to double-mutant epistasis prediction error. Panel E (5) explains the Reliability Score Framework distribution with safety categories. Panel F (6) displays the complete 11,400-mutation prospective sequence landscape of Transketolase (TKT) colored by reliability category. Panel G (7) maps the TKT Industrial Fitness distribution. Panel H (8) shows the top 10 prioritized candidate mutations for wet-lab validation.*

---

## 4. Execution & Automated Generation

All nine figures can be compiled automatically using the master runner script:

```bash
# Execute compilation
PYTHONPATH=. python -m src.figures.generate_all
```

To run generation on individual modules:

```bash
# Generate Figure F2 (Discovery)
PYTHONPATH=. python -m src.figures.discovery

# Generate Figure F3 (Validation)
PYTHONPATH=. python -m src.figures.validation

# Generate Figures F6 & F7 (TKT Landscapes)
PYTHONPATH=. python -m src.figures.tkt_landscape
```

---

## 5. Certification & Quality Control Results

The module is certified via automated testing:
- **Technical Accuracy**: Coordinates of plotted points match the underlying source data exactly.
- **Mathematical Consistency**: Binning of points in Figure F6 matches categories in `reliability_scores.parquet`.
- **Reproducibility**: Run successfully in headless environment using Matplotlib `Agg` backend.
- **Quality Control**: Generates both digital resolution (PNG, 300 DPI) and vector graphics (PDF) for all figures.
