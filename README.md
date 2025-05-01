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
| Logistic Regression | One-hot (512bp) | 0.992      | 0.990      |
| Logistic Regression | 4-mer counts   | 0.750      | 0.685     |
| BLAST              | -              | 0.993    | 0.990      |

---

## Key Insights

- Despite the lack of sequence similarity by design, both BLAST and logistic regression with one-hot encoding achieved near-perfect accuracy and macro F1, suggesting that residual signals or functional motifs may still be detectable.
- Logistic regression with 4-mer counts performed moderately well after full training (F1 = 0.685), despite poor generalization in cross-validation (F1 = 0.036).
- BLAST's strong performance was surprising given the data curation style of sequence dissimilarity, there can be  partial alignments or subtle compositional similarities still in the dataset.
- Overall, traditional approaches—when carefully applied—can match or outperform foundation models like ESM in settings like ConvEnz where function is removed from homology.


---

## Usage

###  Requirements

Install dependencies:

```bash
pip install -r requirements.txt
