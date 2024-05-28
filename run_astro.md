## **File description**

The repository cointains the following script:
1) NAc Locations
 - [Classification of NAc Location](https://github.com/MichelaMarini/Morphological-analysis-of-astrocytes/blob/main/classification_anatomical_location.py): single-cell classification of NAc locations (AC, DM, LAT, VM, PC) based on Random Forest
 - [Statistical analysis](https://github.com/MichelaMarini/Morphological-analysis-of-astrocytes/blob/main/statistical_analysis_NAc_Location.py): ANOVA and Turkey HSD test on single-cell features
- [Morphological analysis](https://github.com/MichelaMarini/Morphological-analysis-of-astrocytes/blob/main/morphological_analysis_NAc_location.py). Plot of the distribution of classes for each shape feature, the EMD distances between NAc locations for each shape feature and the aggregated EMD.
2) Drug use and natural reward
  - [Classification of stages of addictions](https://github.com/MichelaMarini/Morphological-analysis-of-astrocytes/blob/main/classification_drug_use_natural_reward.py): single-cell classification of addiction-related behavior (control, withdrawal relapse) based on Random Forest
  - [Statistical analysis](https://github.com/MichelaMarini/Morphological-analysis-of-astrocytes/blob/main/statistical_analysis_drug_use_natural_reward.py): ANOVA and Turkey HSD test on single-cell features
  - [Morphological analysis](https://github.com/MichelaMarini/Morphological-analysis-of-astrocytes/blob/main/morphological_analysis_drug_use_natural_reward.py): Plot of the distribution of classes for each shape feature, the EMD distances between stages of addiction for each shape feature and the aggregated EMD.

The repository cointains the following data: 
1) [Features](https://github.com/MichelaMarini/Morphological-analysis-of-astrocytes/tree/main/r): cvs files cointaining all the extracted single-cell features 
2) [Original data](https://github.com/MichelaMarini/Morphological-analysis-of-astrocytes/releases/tag/v.1.0.0)

## **Running on Colab**
A Google Colab version is provided as a demo
[![Open Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/MichelaMarini/Morphological-analysis-of-astrocytes/blob/main/astrocytes_demo.ipynb)
