# OptEmbed
OptEmbed is a novel and organism-agnostic ensemble-based framework that leverages embeddings from a protein language model along with sequential features to predict enzyme optimal temperature, melting temperature, and pH level. OptEmbed outperforms current state of the art models in all of the prediction tasks significantly. 

### Data availability
All training and independent datasets are given in [Dataset](Dataset) folder

### Environments
OS: Pop!_OS 22.04 LTS


Python version: Python 3.9.19


Used libraries: 
```
numpy==1.26.4
pandas==2.2.1
xgboost==2.0.3
pickle5==0.0.11
scikit-learn==1.2.2
matplotlib==3.8.2
PyQt5==5.15.10
imblearn==0.0
skops==0.9.0
shap==0.45.1
IPython==8.18.1
pytorch==2.2.2
```

### Reproduce results
1. Firstly, all features can be downloaded from the link given in the [Features/readme.txt](Features/readme.txt) file.

2. In [Training](Training) and [Testing](Testing), reproducable codes are given. Training scripts are also provided.

### Reproduce previous paper metrics
In [prev_papers](prev_papers), scripts are provided for reproducing the results of the previous papers.
