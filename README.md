# 🧠 ClinGRAD - Clinically-Guided GNN for Dementia Diagnosis

## 📌 Overview

ClinGRAD is a clinically-guided heterogeneous graph neural network that combines genomic and radiomics data for accurate dementia diagnosis. It classifies patients into four categories: **Control (CTL), Mild Cognitive Impairment (MCI), Alzheimer's Disease (AD), and Vascular Dementia (VaD)** with state-of-the-art accuracy.

## 🏗️ Architecture

![Model Architecture](./Figures/ClinGRAD_Arch.png)

The framework consists of four key components:

1. 🧠 **Multimodal Fusion**: Integrates MRI-derived radiomics features and genetic data using biologically validated connections.
2. 🔬 **Multi-Scale Representation**: Captures both local brain structure and global genomic pathway relationships.
3. 🔍 **Clinical Interpretability**: Provides attention-based explanations for gene-structure interactions.
4. 🧬 **Pathway-Based Gene Clustering**: Reveals underlying biological mechanisms through unsupervised clustering.

## 🛠 Installation

Ensure you have the required dependencies installed:

```python
# Create Environment
conda create -n clingrad python=3.9
# Activate Environment
conda activate clingrad
# Clone Repo
git clone https://github.com/salmasoma/ClinGRAD/
cd ClinGRAD
# Install requirements
pip install -r requirements.txt
```

# 🚀 Usage

This repository contains the official inference code for the ClinGRAD (Clinically-Guided Genomics and Radiomics Interpretable GNN for Dementia Diagnosis) framework.

### Run inference on new patient data:

```python
python inference.py \
  --model_path "models/clingrad_model.pth" \
  --patient_data "data/new_patients.csv" \
  --gene_data "data/gene_expression.csv" \
  --radiomics_data "data/radiomics_features.csv" \
  --structure_distance_file "data/structure_distance.csv" \
  --dwi_matrix_file "data/dwi_connectivity.csv" \
  --co_expression_file "data/gene_coexpression.csv" \
  --output_path "results/predictions.csv" \
  --modality "RG" \
  --binary \
  --struct \
  --coexp
```

### Using only radiomics data:

```python
python inference.py \
  --model_path "models/clingrad_model.pth" \
  --patient_data "data/new_patients.csv" \
  --radiomics_data "data/radiomics_features.csv" \
  --structure_distance_file "data/structure_distance.csv" \
  --dwi_matrix_file "data/dwi_connectivity.csv" \
  --output_path "results/predictions_R.csv" \
  --modality "R" \
  --struct
```

### Command line arguments:

* `--model_path`: Path to the saved model (required)
* `--patient_data`: Path to patient data CSV (required)
* `--gene_data`: Path to gene data CSV (required for G modality)
* `--radiomics_data`: Path to radiomics data CSV (required for R modality)
* `--structure_distance_file`: Path to structure distance file
* `--dwi_matrix_file`: Path to DWI connectivity matrix file
* `--co_expression_file`: Path to gene co-expression file
* `--modality`: Modalities to use (R=radiomics, G=gene)
* `--struct` / `--no-struct`: Include/exclude structure-to-structure connections
* `--coexp` / `--no-coexp`: Include/exclude gene co-expression connections
* `--binary` / `--multiclass`: Use binary or multiclass classification
* `--heads`: Number of attention heads (default: 4)

## 📊 Performance

ClinGRAD achieves remarkable accuracy across different classification tasks:

* 🔹 AD vs. CTL: 98.75% accuracy
* 🔹 AD vs. MCI: 94.25% accuracy
* 🔹 MCI vs. CTL: 89.66% accuracy
* 🔹 AD vs. VaD: 89.45% accuracy

## 📂 Required Data Files

The model requires the following data files:

* 📋  **Patient Data** : Demographic and clinical information
* 🧬  **Gene Data** : Expression profiles of AD-associated genes
* 🔬  **Radiomics Data** : Structural features extracted from MRI scans
* 🔗  **Structure Distance** : Distance between brain structures
* 🌐  **DWI Matrix** : Brain connectivity patterns from diffusion-weighted imaging
* 🧪  **Gene Co-expression** : Functional relationships between genes
