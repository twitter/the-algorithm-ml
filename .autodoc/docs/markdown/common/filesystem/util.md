[View code on GitHub](https://github.com/twitter/the-algorithm-ml/blob/master/common/filesystem/util.py)

The code in this file provides utility functions for interacting with different file systems, specifically Google Cloud Storage (GCS) and the local file system. It imports the `LocalFileSystem` class from the `fsspec.implementations.local` module and the `gcsfs` module for working with GCS.

Two global variables are defined: `GCS_FS` and `LOCAL_FS`, which are instances of `gcsfs.GCSFileSystem` and `fsspec.implementations.local.LocalFileSystem`, respectively. These instances are used to interact with GCS and the local file system.

The `infer_fs(path: str)` function takes a file path as input and returns the appropriate file system instance based on the path's prefix. If the path starts with "gs://", it returns the `GCS_FS` instance; if it starts with "hdfs://", it raises a `NotImplementedError` as HDFS support is not yet implemented; otherwise, it returns the `LOCAL_FS` instance.

The `is_local_fs(fs)` and `is_gcs_fs(fs)` functions are helper functions that check if the given file system instance is a local file system or a GCS file system, respectively. They return a boolean value based on the comparison.

These utility functions can be used in the larger project to abstract away the details of working with different file systems. For example, when reading or writing data, the project can use the `infer_fs` function to determine the appropriate file system to use based on the input path, and then use the returned file system instance to perform the desired operation.

```python
fs = infer_fs(file_path)
if is_local_fs(fs):
    # Perform local file system operations
elif is_gcs_fs(fs):
    # Perform GCS file system operations
```

This approach allows the project to easily support multiple file systems without having to modify the core logic for each new file system added.
## Questions: 
 1. **Question:** What is the purpose of the `infer_fs` function and how does it determine which file system to use?

   **Answer:** The `infer_fs` function is used to determine the appropriate file system to use based on the given path. It checks the path's prefix to decide whether to use Google Cloud Storage (GCS) file system, Hadoop Distributed File System (HDFS), or the local file system.

2. **Question:** How can HDFS support be added to this code?

   **Answer:** To add HDFS support, you can use the `pyarrow` library's HDFS implementation. Replace the `raise NotImplementedError("HDFS not yet supported")` line with the appropriate code to initialize and return an HDFS file system object.

3. **Question:** What are the `is_local_fs` and `is_gcs_fs` functions used for?

   **Answer:** The `is_local_fs` and `is_gcs_fs` functions are utility functions that check if the given file system object is a local file system or a Google Cloud Storage file system, respectively. They return a boolean value indicating whether the input file system matches the expected type.