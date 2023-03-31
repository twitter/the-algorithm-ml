Twhin in torchrec

Workflow
========
- Build local development images `./scripts/build_images.sh`
- Run with `./scripts/docker_run.sh`
- Iterate in image with `./scripts/idocker.sh`
- Run tests with `./scripts/docker_test.sh`


Dump twhin user follows author prod data to gcs parquet in index-coordinates
===========
## Dump twhin user follows author prod data to gcs parquet in index-coordinates
Assumes the graph has a single partition

```
EXPORT DATA
  OPTIONS ( uri = 'gs://follows_tml_01/tweet_eng/2023-01-23/small/edges/0/*',
    format = 'PARQUET', compression="ZSTD",
    overwrite = TRUE) AS ( (
    SELECT
      lhs, rhs, rel
    FROM
      `twttr-twhin-stg.tweet_eng_tml_01.tweet_eng_small_graph`
    WHERE
      snapshot_date = "2023-01-23" AND rel = 0)
  )
```
