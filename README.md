# BayesTME: A reference-free Bayesian method for analyzing spatial transcriptomics data

This package implements BayesTME, a fully Bayesian method for analyzing ST data without needing single-cell RNA-seq (scRNA) reference data.

## Tutorial

See the file `demo.ipynb` for a tutorial on preprocessing real data to remove technical error with the BayesTME anisotropic correction and generating the K folds for cross-validation. BayesTME uses cross-validation to select the number of cell types. We recommend you run each fold setting separately (e.g. using a compute cluster).

Additional demos on running the deconvolution and spatial transcriptional program code are coming soon!

