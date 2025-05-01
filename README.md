# Convergent Enzyme Classification: Traditional Approaches vs. Foundation Models

## Authors
- Mahbuba Tasmin
- Bryn Reimer

_Course: CS 690U, University of Massachusetts Amherst (Spring 2025)_

## Project Overview

This project evaluates the effectiveness of traditional bioinformatics techniques on the **Convergent Enzyme Classification** task from the **DGEB benchmark**. We specifically assess whether simple modeling pipelines — such as **BLAST**-based nearest neighbor search and **logistic regression** using basic encodings — can match or exceed the performance of modern foundation models like ESM, which tend to underperform due to high sequence divergence in this task.

---

##  Biological Motivation

The `convergent_enzymes` dataset comprises **DNA sequences of enzymes** annotated with **Enzyme Commission (EC) numbers**, where **training and test sequences with the same EC number share little to no sequence similarity**. This simulates convergent evolution, where similar function has evolved independently multiple times. Because of this, even foundation models struggle to generalize — making this an ideal case for re-evaluating traditional approaches.

---

##  Dataset

- Source: [tattabio/ec_classification_dna](https://huggingface.co/datasets/tattabio/ec_classification_dna) (via Hugging Face Datasets)
- Train set: 640 DNA sequences
- Test set: 128 DNA sequences
- Task: Predict EC number (multi-class classification)

---

## Methods

### 🔹 Traditional Machine Learning (scikit-learn)
- **Inputs**: DNA sequences truncated to 512 bp
- **Encodings**:
  - One-hot encoding (512 x 4 = 2048 features)
  - 4-mer count features (256 features)
- **Model**: Multinomial Logistic Regression (L2-regularized)
- **Evaluation**:
  - Accuracy
  - Macro F1-score
  - 5-fold cross-validation

### 🔹 BLAST-based Classification
- Use `blastn` to search test sequences against the training set
- Predict EC label based on the top BLAST hit
- Evaluate using macro F1 and accuracy

---

##  Results Summary

| Model             | Encoding       | Accuracy | Macro F1 |
|------------------|----------------|----------|----------|
| Logistic Regression | One-hot (512bp) | TBD      | TBD      |
| Logistic Regression | 4-mer counts   | TBD      | TBD      |
| BLAST              | -              | TBD      | TBD      |

_(Results will be filled after final runs)_

---

##  Key Insights

- Despite the high divergence in sequence space, logistic regression on interpretable features may still identify functional signals.
- BLAST is expected to struggle by design (no shared alignment hits), but it provides a useful lower bound.
- Comparing performance against DGEB's reported ESM-based models allows us to critically evaluate how well foundation models generalize under function-without-homology scenarios.

---

## Usage

###  Requirements

Install dependencies:

```bash
pip install -r requirements.txt
