# ExamPle

### Introduction

This repository contains code for "ExamPle: Explainable deep learning framework for the prediction of plant small secreted peptides".

### Paper Abstract

Plant Small Secreted Peptides (SSPs) play an important role in plant growth, development, and plant-microbe interactions. Therefore, the identification of SSPs is essential for revealing the functional mechanisms. Over the last few decades, machine learning-based methods have been developed, accelerating the discovery of SSPs to some extent. However, existing methods highly depend on hand-crafted feature engineering, which easily ignores the latent feature representations and impacts the predictive performance. Here, we propose ExamPle, a novel deep learning model using Siamese network and multi-view representation for the explainable prediction of the plant SSPs. Benchmarking comparison results show that our ExamPle performs significantly better than existing methods in the prediction of plant SSPs. Also, our model shows excellent feature extraction ability by using dimension reduction tools. Importantly, by utilizing in silico mutagenesis (ISM) experiments, ExamPle can discover sequence characteristics and identify the contribution of each amino acid. The key novel principle learned by our model is that the head region of the peptide and some specific sequential patterns are strongly associated with the SSPs’ functions. Thus, ExamPle is a competitive model and tool for plant SSPs prediction and effective plant SSPs design.

### Dataset

The dataset in paper "ExamPle" is included in `dataset/SSP_dataset.csv`.

### Usage

```
python main.py
```

### Acknowledgement

Thanks to Hao Cheng (He used to be a member of Weilab and now continues his PhD life in the Ohio State University). He provided some advice and guidance on building the ExamPle framework.
