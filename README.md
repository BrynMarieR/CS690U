# Convergent Enzyme Classification: Traditional Approaches vs. Foundation Models

## Authors

- Mahbuba Tasmin
- Bryn Reimer

_Course: CS 690U, University of Massachusetts Amherst (Spring 2025)_


## Project Overview

This project evaluates the effectiveness of traditional bioinformatics
techniques on the **Convergent Enzyme Classification** task from the **DGEB
benchmark**. We specifically assess how simple modeling pipelines — such as
**BLAST**-based nearest neighbor search and **logistic regression** using basic
encodings — compare to the performance of modern foundation models like
ESM.

---

## Biological Motivation

The `convergent_enzymes` dataset comprises **protein sequences of enzymes**
annotated with **Enzyme Commission (EC) numbers**, where **training and test
sequences with the same EC number share little to no sequence similarity**. This
simulates convergent evolution, where similar function has evolved independently
multiple times. Because of this, even foundation models struggle to generalize —
making this an ideal case for re-evaluating traditional approaches.

---

## Dataset

- Source: [TattaBio Convergent Enzymes](hf://datasets/tattabio/convergent_enzymes/)
- Train set: 2000 amino acid sequences
- Test set: 400 amino acid sequences
- Task: Predict EC number (multi-class classification)

---

## Methods

### Traditional Machine Learning (PyTorch & scikit-learn)
- **Inputs**: Amino acid sequences 
- **Encodings**:
  - One-hot encoding (e.g., max-len x 21 )
  - k-mer count vectors 
  - Count encoding
  - Amino acid properties encoding
- **Model**: Multinomial Logistic Regression (implemented in PyTorch)
- **Evaluation**:
    - Accuracy
    - Macro F1-score
    - Number correct


### BLAST-based Classification

- Use `blastp` to search test sequences against the training set
- Predict EC label based on the top BLAST hit
- Evaluate using accuracy, macro F1, and number correct


---

## Usage

All results available in the main Jupyter notebook