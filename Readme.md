---
# GCN & MAML Pipeline for Document-Level Named Entity Recognition (NER) and Relation Extraction (RE)

This project implements a **Graph Convolutional Network (GCN)** enhanced with **Model-Agnostic Meta-Learning (MAML)** to perform document-level NER and RE. It processes documents into graphs, applies BERT-based embeddings, trains using GCNs, and then fine-tunes the model with MAML to better generalize across domains.
---

## Project Structure

```
project/
├── /data/
│   ├── train/                         # Folder containing JSON files for training
│   ├── test/                          # Folder containing subfolders with test JSON files
│   └── results/                       # Folder where predictions and evaluation results will be saved
```

---

## Model Overview

- **GCN** processes BERT-based node features and syntactic dependency edges from spaCy.
- **NER** is performed by classifying each token into entity types like `PERSON`, `ORG`, `DATE`, etc.
- **RE** is performed by pairing predicted entity spans and classifying relationships.
- **MAML** is applied to learn a domain-agnostic initialization for fast adaptation to new domains.

---

## Dependencies

Install the required packages with:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install spacy transformers
python -m spacy download en_core_web_sm
```

Ensure that your system supports CUDA if running on GPU.

---

## How to Use

### 1. Prepare your data

Put your training and testing files in the following folders:

```
data/train/        # Each file is a list of JSON documents
data/test/         # Supports nested subfolders for evaluation
```

Each document JSON must include:

- `"doc"` or `"document"` (text)
- `"entities"` (with type and mentions)
- `"triples"` (with head, tail, relation)

### 2. Run the pipeline

Step 1: Run GCN.ipynb

This builds graph data and meta-training tasks and saves them as:

data/graphs.pkl

data/train_tasks.pkl

Step 2: Run MAML.ipynb

This loads the graphs and tasks, trains the GCN with MAML, annotates test files and evaluates NER and RE predictions.

---

## Evaluation

Each test JSON should include:

- `"NER-label_set"`
- `"RE_label_set"`

The pipeline uses these to compute:

- **NER Precision / Recall / F1**
- **RE Precision / Recall / F1**
- **Combined macro-averaged scores**

Output structure in results:

```json
{
  "pred_entities": [ { "text": ..., "label": ..., ... } ],
  "pred_triples":  [ { "head": ..., "tail": ..., "label": ..., "conf": ... } ],
  "NER-label_set": [...],
  "RE_label_set":  [...]
}
```

---

## Future Work

- Introduce **reinforcement learning** to iteratively refine RE predictions
- Use **meta-learned optimizers** for task-aware weight updates
- Add **domain-specific adapters** for more robust generalization across diverse corpora

---
