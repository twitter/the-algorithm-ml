[View code on GitHub](https://github.com/twitter/the-algorithm-ml/blob/master/tools/pq.py)

The code in this file provides a local reader for parquet files, which are a columnar storage file format optimized for use with big data processing frameworks. The main class, `PqReader`, is designed to read parquet files and perform various operations on the data, such as displaying the first few rows, displaying unique values in specified columns, and showing the schema of the dataset.

The `_create_dataset` function is a helper function that takes a file path as input and returns a PyArrow dataset object. This dataset object is used by the `PqReader` class to perform operations on the data.

The `PqReader` class has an `__init__` method that initializes the dataset, batch size, number of rows to read, and columns to read. The `__iter__` method allows the class to be used as an iterator, yielding batches of data from the dataset. The `_head` method reads the first `--num` rows of the dataset, while the `bytes_per_row` property calculates the approximate size of each row in bytes.

The `schema` method prints the schema of the dataset, which describes the structure and types of the columns. The `head` method displays the first `--num` rows of the dataset, while the `distinct` method displays unique values seen in specified columns in the first `--num` rows. This can be useful for getting an approximate vocabulary for certain columns.

The code also provides examples of how to use the `PqReader` class with command-line arguments, such as displaying the first few rows of a dataset or showing the distinct values in specified columns.

For example, to display the first 5 rows of a dataset, you can use the following command:

```
python3 tools/pq.py \
  --num 5 --path "tweet_eng/small/edges/all/*" \
  head
```

To display the distinct values in the "rel" column, you can use the following command:

```
python3 tools/pq.py \
  --num 1000000000 --columns '["rel"]' \
  --path "tweet_eng/small/edges/all/*" \
  distinct
```
## Questions: 
 1. **Question**: What is the purpose of the `PqReader` class and its methods?
   **Answer**: The `PqReader` class is designed to read parquet files locally and provide functionality to display the first `--num` rows of the dataset (`head` method), display the schema of the dataset (`schema` method), and display unique values seen in specified columns in the first `--num` rows (`distinct` method).

2. **Question**: How does the `_create_dataset` function work and what does it return?
   **Answer**: The `_create_dataset` function takes a file path as input, infers the filesystem using the `infer_fs` function, and then creates a list of files using the `fs.glob` method. It returns a pyarrow dataset object with the specified format (parquet) and filesystem.

3. **Question**: What is the purpose of the `bytes_per_row` property in the `PqReader` class?
   **Answer**: The `bytes_per_row` property calculates the estimated size of a row in bytes based on the bit width of each column in the dataset schema. This is used to check if the total bytes to be read in the `_head` method exceed a certain limit (500 MB) to prevent excessive memory usage.