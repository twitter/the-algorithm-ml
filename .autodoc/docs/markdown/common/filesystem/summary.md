[View code on GitHub](https://github.com/twitter/the-algorithm-ml/tree/master/.autodoc/docs/json/common/filesystem)

The code in the `filesystem` folder provides utility functions for handling file system operations in the larger machine learning project. It allows the project to seamlessly work with different types of file systems, such as local storage or Google Cloud Storage (GCS), by abstracting away the details of interacting with these file systems.

The `__init__.py` file imports three utility functions from the `tml.common.filesystem.util` module:

1. `infer_fs(file_path)`: Determines the type of file system based on the given file path. It returns an instance of the appropriate file system class, either local or GCS.

   ```python
   file_path = "gs://my-bucket/data.csv"
   fs = infer_fs(file_path)
   ```

2. `is_gcs_fs(fs)`: Checks if the given file system object is a GCS file system. Returns `True` if it is, and `False` otherwise.

   ```python
   file_path = "gs://my-bucket/data.csv"
   fs = infer_fs(file_path)
   if is_gcs_fs(fs):
       # Perform GCS-specific operations
   ```

3. `is_local_fs(fs)`: Checks if the given file system object is a local file system. Returns `True` if it is, and `False` otherwise.

   ```python
   file_path = "/home/user/data.csv"
   fs = infer_fs(file_path)
   if is_local_fs(fs):
       # Perform local file system-specific operations
   ```

The `util.py` file provides the implementation of these utility functions. It defines two global variables, `GCS_FS` and `LOCAL_FS`, which are instances of `gcsfs.GCSFileSystem` and `fsspec.implementations.local.LocalFileSystem`, respectively. These instances are used to interact with GCS and the local file system.

These utility functions can be used in the larger project to abstract away the details of working with different file systems. For example, when reading or writing data, the project can use the `infer_fs` function to determine the appropriate file system to use based on the input path, and then use the returned file system instance to perform the desired operation.

```python
fs = infer_fs(file_path)
if is_local_fs(fs):
    # Perform local file system operations
elif is_gcs_fs(fs):
    # Perform GCS file system operations
```

This approach allows the project to easily support multiple file systems without having to modify the core logic for each new file system added.
