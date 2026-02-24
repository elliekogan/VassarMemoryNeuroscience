# Vassar College Memory Neuroscience Lab

Computational and behavioral analysis resources for memory neuroscience research at Vassar College.

## Research Focus

The Memory Neuroscience Lab investigates how memory is encoded, stored, and retrieved across molecular, systems, and behavioral levels. A central emphasis is fear-memory generalization and the neural mechanisms that support adaptive versus maladaptive generalization, with translational relevance to anxiety-related disorders and post-traumatic stress disorder (PTSD).

## Repository Scope

This repository provides lab-maintained analysis workflows used for behavioral video data processing and downstream quantification.

- `DeepLabCut_Extensions/`: analysis notebooks and scripts extending DeepLabCut outputs.
- `VAME/`: resources for integrating and extending VAME-based workflows.

## Getting Started

1. Clone this repository.
2. Create and activate a dedicated Python environment.
3. Install required dependencies for the workflow(s) you plan to run.
4. Configure project paths using your local data directory structure.
5. Run analyses from the relevant module README.

## Data and Path Configuration

To ensure reproducibility across systems:

- Do not hard-code machine-specific paths.
- Use configurable variables (for example, `INPUT_DIR`, `OUTPUT_DIR`, and `PROJECT_ROOT`).
- Keep raw data outside version control and document expected file formats.

## Reproducibility Standards

- Record software versions (Python, package versions, model versions).
- Keep analysis parameters explicit and versioned.
- Export derived results to structured tables (`.csv`) for downstream statistics.

## Funding

The lab's research program is supported by the National Institute of Mental Health (NIMH), award `R15 MH127534-01`.
