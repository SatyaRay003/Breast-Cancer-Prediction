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
|ML Model Name|Accuracy|Precision|Recall|F1 Score|AUC Score|MCC Score|
|---|---|---|---|---|---|---|
|Logistic Regression|0\.9737|0\.9762|0\.9535|0\.9647|0\.9697|0\.9439|
|Decision Tree|0\.9474|0\.9302|0\.9302|0\.9302|0\.9440|0\.8880|
|K-Nearest Neighbor|0\.9474|0\.9302|0\.9302|0\.9302|0\.9440|0\.8880|
|Gaussian Naive Bayes|0\.9649|0\.9756|0\.9302|0\.9524|0\.9581|0\.9253|
|Random Forest|0\.9649|0\.9756|0\.9302|0\.9524|0\.9581|0\.9253|
|XG Boost|0\.9561|0\.9524|0\.9302|0\.9412|0\.9510|0\.9064|


## Observations of Models Performance:

- Breast cancer prediction is a medical diagnosis. Missing a positive case (false negatives) 
can lead to severe consequences for a patient. For this reason, to reduce the risk of overlooking serious cancer issues,
we should consider **Recall** as our primary metric to compare the performance of different machine learning models.


|ML Model Name| Observation about model performance                                                        |
|---|--------------------------------------------------------------------------------------------|
|Logistic Regression| Amounf all the model, Logistic Regression model performed best on Recall value with 0.9535 |
|Decision Tree|                                                                                   |
|K-Nearest Neighbor|                                                                                    |
|Gaussian Naive Bayes|                                                                               |
|Random Forest|                                                                               |
|XG Boost|                                                                                   |





