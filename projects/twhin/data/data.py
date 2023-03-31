from tml.projects.twhin.data.config import TwhinDataConfig
from tml.projects.twhin.models.config import TwhinModelConfig
from tml.projects.twhin.data.edges import EdgesDataset


def create_dataset(data_config: TwhinDataConfig, model_config: TwhinModelConfig):
  tables = model_config.embeddings.tables
  table_sizes = {table.name: table.num_embeddings for table in tables}
  relations = model_config.relations

  pos_batch_size = data_config.per_replica_batch_size
  global_negatives = data_config.global_negatives
  in_batch_negatives = data_config.in_batch_negatives

  return EdgesDataset(
    file_pattern=data_config.data_root,
    relations=relations,
    table_sizes=table_sizes,
    global_negatives=global_negatives,
    in_batch_negatives=in_batch_negatives,
    batch_size=pos_batch_size,
  )
