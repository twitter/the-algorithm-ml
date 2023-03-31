[View code on GitHub](https://github.com/twitter/the-algorithm-ml/blob/master/ml_logging/absl_logging.py)

This code sets up logging for the `the-algorithm-ml` project using the `absl` (Abseil) library. The primary purpose of this code is to configure the logging system to output logs to `sys.stdout` instead of the default `sys.stderr`. This is done to ensure that severity levels in Google Cloud Platform (GCP) Stackdriver are accurate.

The code defines a function `setup_absl_logging()` that configures the logging system. Inside the function, the `absl` logging handler's stream is set to `sys.stdout` to redirect logs to the standard output. A custom log formatter is also defined using the `py_logging.Formatter` class, which formats log messages with the module name, function name, line number, log level, and the actual log message. The formatter is then set for the `absl` logging handler. Finally, the logging verbosity level is set to `logging.INFO` to display log messages with a severity level of `INFO` and higher.

After defining the `setup_absl_logging()` function, it is called immediately to configure the logging system for the project. To use this logging configuration in other parts of the project, the following example demonstrates how to import and use the configured `logging`:

```python
from twitter.ml.logging.absl_logging import logging
logging.info(f"Properly logged as INFO level in GCP Stackdriver.")
```

By using this logging setup, developers can ensure that log messages are properly formatted and directed to the correct output stream, making it easier to monitor and debug the project using GCP Stackdriver.
## Questions: 
 1. **Question:** What is the purpose of the `setup_absl_logging()` function in this code?

   **Answer:** The `setup_absl_logging()` function is used to configure the absl logging library to push logs to stdout instead of stderr and to set a custom log message format.

2. **Question:** How can a developer change the log severity level in this code?

   **Answer:** The developer can change the log severity level by modifying the `logging.set_verbosity(logging.INFO)` line in the `setup_absl_logging()` function, replacing `logging.INFO` with the desired log level (e.g., `logging.DEBUG`, `logging.WARNING`, etc.).

3. **Question:** What is the custom log message format used in this code?

   **Answer:** The custom log message format used in this code is `"[%(module)s.%(funcName)s:%(lineno)s - %(levelname)s] %(message)s"`, which includes the module name, function name, line number, log level, and the log message itself.