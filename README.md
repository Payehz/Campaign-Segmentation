![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)

# Customer Campaign Segmentation
The objective of the model is to predict the customers' segmentation base of some features.

## Results

The model scores an astounding accuracy of 92%, it is neither overfitting nor underfitting.

![Score](static/classification_report.png)

TensorBoard was used to visualise the results. Here is the epochs graph.

![TensorBoard](static/tensorboard.png)

## Model architecture

The model used to train this data consists of two hidden layers, excluding input and output.
Each hidden layers contains 64 nodes, after each layer it goes through batch normalization and dropout value of 20%.

![Model](static/model.png)

# Credits

The data is downloaded from
[Kaggle](https://www.kaggle.com/datasets/kunalgupta2616/hackerearth-customer-segmentation-hackathon)




