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

[numpy](https://github.com/numpy/numpy)>=1.20.0 <br>
[scipy](https://github.com/scipy/scipy) <br>
[scikit-image](https://github.com/scikit-image/scikit-image)>=0.20.0 <br>
[scikit-learn](https://github.com/scikit-learn/scikit-learn) <br>
[pandas](https://github.com/pandas-dev/pandas)>=2.0.0 <br>
[seaborn](https://github.com/mwaskom/seaborn) <br>
[xlrd](https://github.com/python-excel/xlrd) <br>
[jupyter](https://github.com/jupyter/notebook) <br>
[matplotlib](https://github.com/matplotlib/matplotlib) <br>
[pathlib](https://github.com/budlight/pathlib) <br>

### Environment installation

```
conda create --name=astro
conda activate astro
conda install h5py jupyter matplotlib 
conda install scipy
conda install scikit-image scikit-learn 
conda install seaborn matplotlib
conda install xlrd pathlib pandas
```

### Local installation

To install the code locally on your computer and in order to be able to edit it, first download it from github to your computer, go to one directory above where your code folder is
downloaded/located/saved, through an anaconda3 terminal, the structure of the directory would look like this:
```
dir/
 Astro/
      setup.py
      README.md
```
type ```pip install -e Astro``` in the same terminal. The ```-e``` allows one to edit the program.
  
All the required packages will be installed automatically from ``` setup.py ``` file.

### Github installation
To install it directly from github:
    ``` pip install git+https://github.com/MichelaMarini/Morphological-analysis-of-astrocytes.git ```
