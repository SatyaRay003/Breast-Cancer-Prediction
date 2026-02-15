## Problem Statement:

Breast Cancer Prediction; with the numerical features extracted from cell nucleus image, 
We have to predict whether the tumor is benign or malignant.


## Dataset description:

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

- Target attribute: Diagnosis (M = malignant, B = benign)

With the ten real-valued features computed for each cell nucleus:

- a) radius (mean of distances from center to points on the perimeter)
- b) texture (standard deviation of gray-scale values)
- c) perimeter
- d) area
- e) smoothness (local variation in radius lengths)
- f) compactness (perimeter^2 / area - 1.0)
- g) concavity (severity of concave portions of the contour)
- h) concave points (number of concave portions of the contour)
- i) symmetry
- j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features.

## Models used:
| ML Model Name            |Accuracy|AUC Score|Precision|Recall|F1 Score|MCC Score|
|--------------------------|---|---|---|---|---|---|
| Logistic Regression      |0\.9649|0\.9638|0\.9574|0\.9574|0\.9574|0\.9276|
| Decision Tree            |0\.9123|0\.9158|0\.8627|0\.9362|0\.8980|0\.8234|
| K-Nearest Neighbour      |0\.9561|0\.9468|1\.0000|0\.8936|0\.9438|0\.9119|
| Gaussian Naive Bayes     |0\.9298|0\.9276|0\.9149|0\.9149|0\.9149|0\.8552|
| Random Forest (Ensemble) |0\.9737|0\.9744|0\.9583|0\.9787|0\.9684|0\.9460|
| XG Boost (Ensemble)      |0\.9825|0\.9819|0\.9787|0\.9787|0\.9787|0\.9638|


## Observations of Models Performance:

- Breast cancer prediction is a medical diagnosis. Missing a positive case (false negatives) 
can lead to severe consequences for a patient. For this reason, to reduce the risk of overlooking serious cancer issues,
we should consider **Recall** as our primary metric to compare the performance of different machine learning models.


| ML Model Name            | Observation about model performance                                                                                                                                                                                 |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression      | The Logistic Regression model performed significantly good with 0.9574 Recall value. In other evaluation metrics also, it performed well making it 3rd best model among all.                                        |
| Decision Tree            | The Decision Tree model performed average in terms of Recall value (0.9362) comparing with other models. But it completed training within 0.0474 seconds making it 2nd fastest model with respect to training time. |
| K-Nearest Neighbour      | The K-Nearest Neighbour model performed worst among all the models with 0.8936 Recall value.                                                                                                                        |
| Gaussian Naive Bayes     | The Gaussian Naive Bayes performed below average as compared to other machine learning models but completed training fastest within 0.0040 seconds.                                                                 |
| Random Forest (Ensemble) | The Random Forest model performed best in Recall Value (0.9787) among all the models and 2nd best with respect to all the evaluation metrics.                                                                       |
| XG Boost (Ensemble)      | The XG Boost model performed best among all, consistently over all the evaluation metrics and specifically with 0.9787 Recall value making it most preferable for medical diagnosis of Breast Cancer detection.     |





