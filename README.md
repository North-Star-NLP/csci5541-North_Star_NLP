# Knowledge-Base Poisoning Defense in Retrieval-Augmented Generation

**Authors:** Logan Oakley, Zhixing (Sean) Jiang, Samuel Kekeocha  
**Course:** Spring 2026 CSCI 5541 NLP - University of Minnesota

## Project Overview

This project investigates defense strategies against knowledge-base poisoning attacks in retrieval-augmented generation (RAG) systems. We evaluate multiple defense approaches and measure their effectiveness in reducing attack success rates.

## Project Structure

```
.
├── src/                                    # Source code and notebooks
│   ├── main_RAG.ipynb                     # Main RAG pipeline implementation
│   ├── semantic_poisoning.ipynb           # Experimental evaluation notebook
│   ├── model_function.py                  # Model loading utilities
│   ├── evaluation_function.py             # Evaluation framework
│   ├── helper_function.py                 # Helper utilities
│   ├── queries/
│   │    └── queries.json               # 20 benchmark queries
│   ├── poisoned_data/
│   │     ├── semantic_poisoned_data.json
│   │     └── contradictory_poisoned_data.json
│
├── docs/                                   # Project website
│   └── index.html                         # Interactive documentation
│
└── README.md                               # This file
```

## Key Directories

- **`src/semantic_poisoning.ipynb`** — Main experimental notebook evaluating all defense strategies
- **`src/data/queries/queries.json`** — 20 target queries for evaluation
- **`src/data/poisoned_data/`** — Injected adversarial documents for testing

## Quick Start

1. Review project documentation: `docs/index.html`
2. Run experiments: Open `src/semantic_poisoning.ipynb` in Jupyter
3. Check query details: `src/queries/queries.json`
4. View poisoned documents: `src/poisoned_data/`
