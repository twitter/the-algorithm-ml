[View code on GitHub](https://github.com/twitter/the-algorithm-ml/tree/master/.autodoc/docs/json/common)

The code in the `json/common` folder provides essential functionalities for the `the-algorithm-ml` project, such as implementing the main machine learning algorithm, handling data batches, setting up devices, logging model weights, running distributed training, managing configurations, and integrating with Weights and Biases (WandB). These components work together to enable efficient training and prediction with machine learning models in various applications.

For example, the `MLAlgorithm` class in `__init__.py` provides a high-level interface for training and predicting with a machine learning model. It can be used to train a model on a specific dataset and then use that model to make predictions on new data:

```python
ml_algorithm = MLAlgorithm()
ml_algorithm.train(features, labels)
predictions = ml_algorithm.predict(new_data)
```

The `batch.py` file provides classes for handling data batches in different formats, making it easier to work with various datasets and perform operations such as moving data between devices or pinning memory:

```python
CustomBatch = DataclassBatch.from_fields("CustomBatch", {"field1": torch.Tensor, "field2": torch.Tensor})
batch_gpu = batch.to(torch.device("cuda"))
```

The `device.py` file sets up the appropriate device and backend for running machine learning algorithms on either CPU or GPU, depending on the available resources, ensuring optimal performance and efficient resource utilization.

The `log_weights.py` file logs model weights and embedding table norms during the training process, which can be useful for monitoring the quality of the learned embeddings:

```python
logged_weights = weights_to_log(model, how_to_log=torch.norm)
logged_norms = log_ebc_norms(model_state_dict, ebc_keys, sample_size=4_000_000)
```

The `run_training.py` file serves as a wrapper for single-node, multi-GPU PyTorch training, simplifying the process of setting up and running distributed PyTorch training:

```python
maybe_run_training(
    train_fn,
    "my_module",
    nproc_per_node=4,
    num_nodes=2,
    is_chief=True,
    set_python_path_in_subprocess=True,
    learning_rate=0.001,
    batch_size=64,
)
```

The `utils.py` file provides a function called `setup_configuration` that manages and accesses configuration settings in a structured and type-safe manner:

```python
config, config_path = setup_configuration(MyConfig, "path/to/config.yaml", True)
```

The `wandb.py` file defines a configuration class `WandbConfig` for the Weights and Biases (WandB) integration, ensuring a consistent and organized approach to experiment tracking:

```python
config = WandbConfig(...)
wandb.init(name=config.name, entity=config.entity, project=config.project, tags=config.tags, notes=config.notes, config=config.metadata)
```

These components work together to provide a comprehensive and efficient framework for training and predicting with machine learning models in the `the-algorithm-ml` project.
