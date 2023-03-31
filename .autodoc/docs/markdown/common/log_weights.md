[View code on GitHub](https://github.com/twitter/the-algorithm-ml/blob/master/common/log_weights.py)

The code in this file is responsible for logging model weights and embedding table norms during the training process of a machine learning model in the `the-algorithm-ml` project. It provides two main functions: `weights_to_log` and `log_ebc_norms`.

The `weights_to_log` function takes a PyTorch model and an optional callable or dictionary of callables as input. It traverses the model's parameters and applies the specified function(s) to them. If a single function is provided, it is applied to all parameters. If a dictionary is provided, it applies the corresponding function to the specified parameters. The function returns a dictionary containing the processed weights, which can be logged for monitoring the training process.

Example usage:

```python
model = torch.nn.Linear(10, 5)
logged_weights = weights_to_log(model, how_to_log=torch.norm)
```

The `log_ebc_norms` function logs the norms of the embedding tables specified by `ebc_keys`. It takes the model's state dictionary, a list of embedding keys, and an optional sample size as input. The function computes the average norm per rank for the specified embedding tables and returns a dictionary containing the norms. This can be useful for monitoring the quality of the learned embeddings during training.

Example usage:

```python
model_state_dict = model.state_dict()
ebc_keys = ["model.embeddings.ebc.embedding_bags.meta__user_id.weight"]
logged_norms = log_ebc_norms(model_state_dict, ebc_keys, sample_size=4_000_000)
```

Both functions are designed to work with distributed training, specifically with the `DistributedModelParallel` class from the `torchrec.distributed` module. They use the `torch.distributed` package to gather and log information from all participating devices in the distributed training setup.
## Questions: 
 1. **Question**: What is the purpose of the `how_to_log` parameter in the `weights_to_log` function, and how does it affect the logging of model parameters?
   **Answer**: The `how_to_log` parameter is used to specify how the model parameters should be logged. If it is a function, it will be applied to every parameter in the model. If it is a dictionary, it will only apply and log the specified parameters with their corresponding functions.

2. **Question**: What is the role of the `sample_size` parameter in the `log_ebc_norms` function, and how does it affect the computation of average norms?
   **Answer**: The `sample_size` parameter limits the number of rows per rank to compute the average norm on, in order to avoid out-of-memory (OOM) errors. If you observe frequent OOM errors, you can change the `sample_size` or remove weight logging.

3. **Question**: How does the `log_ebc_norms` function handle the case when an embedding key is not present in the `model_state_dict`?
   **Answer**: If an embedding key is not present in the `model_state_dict`, the function initializes the `norms` tensor with a value of -1 and continues with the next embedding key. This ensures that the missing key does not affect the computation of norms for other keys.