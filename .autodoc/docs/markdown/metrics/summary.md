[View code on GitHub](https://github.com/twitter/the-algorithm-ml/tree/master/.autodoc/docs/json/metrics)

The code in the `json/metrics` folder provides essential evaluation metrics and aggregation methods for assessing the performance of machine learning models and algorithms within the `the-algorithm-ml` project. The folder contains implementations for computing the stable mean, Area Under the Receiver Operating Characteristic (AUROC) curve, and Relative Cross Entropy (RCE) metrics.

The `StableMean` class, found in `aggregation.py`, computes the stable mean of a given set of values using the Welford algorithm. This method is useful for calculating the average performance of a model across multiple runs or datasets, ensuring that the mean is not affected by extreme values or outliers. Example usage:

```python
from the_algorithm_ml.aggregation import StableMean

values = [1, 2, 3, 4, 5]
stable_mean = StableMean()
mean = stable_mean(values)
```

The `AUROCWithMWU` class, found in `auroc.py`, calculates the AUROC curve using the Mann-Whitney U test. This metric is widely used to evaluate the performance of binary classification models, as it measures the trade-off between true positive rate and false positive rate. Example usage:

```python
from the_algorithm_ml.auroc import AUROCWithMWU

true_labels = [0, 1, 0, 1, 1]
predicted_scores = [0.1, 0.8, 0.3, 0.9, 0.6]
auroc = AUROCWithMWU()
score = auroc(true_labels, predicted_scores)
```

The `NRCE` and `RCE` classes, found in `rce.py`, compute the cross-entropy-based metrics for evaluating the performance of multi-class classification models. These metrics are useful for comparing the predicted probabilities of a model against the true class labels. Example usage:

```python
from the_algorithm_ml.rce import NRCE, RCE

true_labels = [0, 1, 2, 1, 0]
predicted_probabilities = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.1, 0.8, 0.1], [0.9, 0.05, 0.05]]
nrce = NRCE()
rce = RCE()
nrce_score = nrce(true_labels, predicted_probabilities)
rce_score = rce(true_labels, predicted_probabilities)
```

In summary, the code in this folder enables users to assess the performance of their machine learning models and algorithms by providing essential evaluation metrics and aggregation methods. These metrics and methods can be easily integrated into the larger project, allowing developers to evaluate and compare different models and algorithms.
