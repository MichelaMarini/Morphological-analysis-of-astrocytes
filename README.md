![image](https://github.com/MichelaMarini/Morphological-analysis-of-astrocytes/assets/70772881/6ec684cb-f86b-4a7d-a6eb-e027bf914746)


## **Purpose and Use Case**
Astrocyte image analysis pipeline to automatically extract and analyze single-cell phenotypic profiles of astrocytes in micrographs. 
The pipeline consist of four processing units, i.e., (1) [astrocyte detection](https://github.com/yewen/AstrocyteDetection), (2) [segmentation](https://github.com/zhaoheng001/Segmentation_code/tree/main)
(3) feature extraction and selection, and (4) [morphological and statistical modeling](https://github.com/MichelaMarini/Morphological-analysis-of-astrocytes)

![Image Description](https://github.com/MichelaMarini/Morphological-analysis-of-astrocytes/blob/main/images/pipeline.jpg)

## **Installation**

### System requirements
Linux, Windows and Mac OS are supported for running the code. At least 8GB of RAM is required to run the software. The software has been heavily tested on Windows 11 and Ubuntu 18.04 and less well-tested on Mac OS. Please open an issue if you have problems with installation.

### Dependencies 
The code relies on the following excellent packages (which are automatically installed with conda/pip if missing):

[numpy](https://github.com/numpy/numpy)
[scipy](https://github.com/scipy/scipy) <br>
[scikit-learn](https://github.com/scikit-learn/scikit-learn) <br>
[pandas](https://github.com/pandas-dev/pandas) <br>
[seaborn](https://github.com/mwaskom/seaborn) <br>
[statsmodels](https://github.com/statsmodels/statsmodels) <br>
[jupyter](https://github.com/jupyter/notebook) <br>
[matplotlib](https://github.com/matplotlib/matplotlib) <br>
[opencv-python](https://github.com/budlight/pathlib) <br>

### Environment installation

```conda``` virtual environment is recommended.
```
conda create -n astro python=3.10
conda activate astro
pip install -r requirements.txt
pip install -e .
```

### Github installation
To install it directly from github:
    ``` pip install git+https://github.com/MichelaMarini/Morphological-analysis-of-astrocytes.git ```
