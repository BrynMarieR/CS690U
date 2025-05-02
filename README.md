# Convergent Enzyme Classification: Traditional Approaches vs. Foundation Models

## Authors

- Mahbuba Tasmin
- Bryn Reimer

_Course: CS 690U, University of Massachusetts Amherst (Spring 2025)_


## Project Overview

This project evaluates the effectiveness of traditional bioinformatics
techniques on the **Convergent Enzyme Classification** task from the **DGEB
benchmark**. We specifically assess whether simple modeling pipelines — such as
**BLAST**-based nearest neighbor search and **logistic regression** using basic
encodings — can match or exceed the performance of modern foundation models like
ESM, which tend to underperform due to high sequence divergence in this task.

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

-
Source: [TattaBio Convergent Enzymes](hf://datasets/tattabio/convergent_enzymes/)
- Train set: 2000 amino acid sequences (protein sequences)
- Test set: 400 amino acid sequences

The `convergent_enzymes` dataset comprises **protein sequences of enzymes** annotated with **Enzyme Commission (EC) numbers**, where **training and test sequences with the same EC number share little to no sequence similarity**. This simulates convergent evolution, where similar function has evolved independently multiple times. Because of this, even foundation models struggle to generalize — making this an ideal case for re-evaluating traditional approaches.

---


##  Dataset

- Source: [tattabio/convergent_enzymes](https://huggingface.co/datasets/tattabio/convergent_enzymes) (via Hugging Face Datasets)
- Train set: 2000 protein sequences
- Test set: 400 protein sequences
- Task: Predict EC number (multi-class classification)

---

## Methods

### Traditional Machine Learning (PyTorch & scikit-learn)
- **Inputs**: Amino acid sequences 
- **Encodings**:
  - One-hot encoding (e.g., max-len x 21 )
  - 3-mer count vectors 
- **Model**: Multinomial Logistic Regression (implemented in PyTorch)
- **Evaluation**:
    - Accuracy
    - Macro F1-score
    - 5-fold cross-validation


### BLAST-based Classification

- Use `blastp` to search test sequences against the training set
- Predict EC label based on the top BLAST hit
- Evaluate using macro F1 and accuracy


---

## Results Summary


| Model                | Encoding         | Accuracy | Macro F1 |
|---------------------|------------------|----------|----------|
| Logistic Regression | One-hot (256 aa) | 1.5%     | 0.011    |
| Logistic Regression | 3-mer counts     | 0.25%    | 0.00003  |
| BLAST               | Top hit only     | 0%       | 0%       |


---

## Key Insights




## Usage

