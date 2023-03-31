[View code on GitHub](https://github.com/twitter/the-algorithm-ml/blob/master/metrics/rce.py)

This code defines two classes, `RCE` and `NRCE`, which compute the Relative Cross Entropy (RCE) and Normalized Relative Cross Entropy (NRCE) metrics, respectively. These metrics are used for evaluating models that predict the probability of success, such as pCTR (predicted Click-Through Rate) models.

The `RCE` class computes the RCE metric by comparing the binary cross entropy of the model to a reference straw man model. The straw man model is a constant predictor, always predicting the average over the labels. The RCE is calculated as:

```
RCE(model) = 100 * (CE(reference model) - CE(model)) / CE(reference model)
```

The `NRCE` class computes the NRCE metric by normalizing the model's predictions to match the average label seen so far. This metric can help identify the potential performance of a well-calibrated model.

Both classes inherit from `torchmetrics.Metric` and implement the `update`, `compute`, and `reset` methods. The `update` method updates the metric state with new predictions and ground truth labels, the `compute` method calculates the accumulated metric, and the `reset` method resets the metric state.

The code also provides utility functions for smoothing values (`_smooth`) and computing binary cross entropy with clipping (`_binary_cross_entropy_with_clipping`). These functions are used internally by the `RCE` and `NRCE` classes.

In the larger project, these classes can be used to evaluate the performance of machine learning models that predict probabilities, such as pCTR models in online advertising. Users can create instances of the `RCE` or `NRCE` classes and update them with model predictions and ground truth labels to compute the accumulated metric.
## Questions: 
 1. **What is the purpose of the `_smooth` function?**

   The `_smooth` function is used to apply label smoothing to the given values. Label smoothing is a technique used to prevent overfitting by adding a small constant to the target labels. This is done by multiplying the value by `(1.0 - label_smoothing)` and adding `0.5 * label_smoothing`.

2. **What is the difference between the `RCE` and `NRCE` classes?**

   The `RCE` class computes the Relative Cross Entropy metric, which measures the performance of a model predicting the probability of success compared to a reference straw man model. The `NRCE` class calculates the RCE of the normalized model, where the normalized model prediction average is normalized to the average label seen so far. The main difference is that NRCE is used to measure how good a model could potentially perform if it was well calibrated, while RCE measures the actual performance of the model.

3. **How does the `update` method work in the `NRCE` class?**

   The `update` method in the `NRCE` class first normalizes the predictions by applying the sigmoid function if the `nrce_from_logits` flag is set to True. Then, it applies label smoothing to the target labels and updates the mean label and mean prediction accumulators. Finally, it normalizes the predictions by multiplying them with the ratio of the mean label to the mean prediction and updates the binary cross entropy accumulator with the computed values.