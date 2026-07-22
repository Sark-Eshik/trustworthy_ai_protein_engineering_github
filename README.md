# Trustworthy AI for Protein Engineering

Discovering Mutation-Space Reliability and Applying It to Industrial Carbon-Capture Enzyme Design.

## Repository Setup and Infrastructure

This repository is built using an expert scientific architecture designed to evaluate whether mutation-space sparsity predicts protein prediction failure.

### Active Directories Tree

1. `configs/`: Centralized environment configurations.
2. `data/`: Raw and processed dataset files.
3. `docs/`: Design and architectural documentation.
4. `notebooks/`: Development and exploratory notebooks.
5. `outputs/`: Intermediary process dumps.
6. `reports/`: Hardware profiles and certifications.
7. `results/`: Process stats and metadata logs.
8. `tests/`: Automated unit and integration test suites.
9. `logs/`: Application execution traces.
10. `checkpoints/`: State serialization files.
11. `models/`: Cached ML model parameters.
12. `scripts/`: Diagnostic and validation scripts.
13. `figures/`: Scientific publication plots.
14. `papers/`: Journal manuscripts.
15. `science_fair/`: Competitive fair board presentations.
16. `presentations/`: Slide decks and pitch scripts.

### Installation

```bash
conda env create -f environment.yml
conda activate trustworthy_ai
```

### Initial Certification

Run the project bootstrap script to verify directory trees, map environment profiles, and verify all structural dependencies:

```bash
cd /Users/ryansarkeshik/Documents/trustworthy_ai_protein_engineering
PYTHONPATH=. /opt/anaconda3/envs/trustworthy_ai/bin/python -m src.bootstrap
```

### Running Tests

Execute the foundational infrastructure tests to confirm config-loading, serialization, dataset-registries, and schema validation engines are fully functional:

```bash
PYTHONPATH=. /opt/anaconda3/envs/trustworthy_ai/bin/pytest tests/infrastructure/
```
