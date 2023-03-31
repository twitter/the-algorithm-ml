import typing
import enum

from tml.common.modules.embedding.config import LargeEmbeddingsConfig
from tml.core.config import base_config
from tml.optimizers.config import OptimizerConfig

import pydantic
from pydantic import validator


class TwhinEmbeddingsConfig(LargeEmbeddingsConfig):
  @validator("tables")
  def embedding_dims_match(cls, tables):
    embedding_dim = tables[0].embedding_dim
    data_type = tables[0].data_type
    for table in tables:
      assert table.embedding_dim == embedding_dim, "Embedding dimensions for all nodes must match."
      assert table.data_type == data_type, "Data types for all nodes must match."
    return tables


class Operator(str, enum.Enum):
  TRANSLATION = "translation"


class Relation(pydantic.BaseModel):
  """graph relationship properties and operator"""

  name: str = pydantic.Field(..., description="Relationship name.")
  lhs: str = pydantic.Field(
    ...,
    description="Name of the entity on the left-hand-side of this relation. Must match a table name.",
  )
  rhs: str = pydantic.Field(
    ...,
    description="Name of the entity on the right-hand-side of this relation. Must match a table name.",
  )
  operator: Operator = pydantic.Field(
    Operator.TRANSLATION, description="Transformation to apply to lhs embedding before dot product."
  )


class TwhinModelConfig(base_config.BaseConfig):
  embeddings: TwhinEmbeddingsConfig
  relations: typing.List[Relation]
  translation_optimizer: OptimizerConfig

  @validator("relations", each_item=True)
  def valid_node_types(cls, relation, values, **kwargs):
    table_names = [table.name for table in values["embeddings"].tables]
    assert relation.lhs in table_names, f"Invalid lhs node type: {relation.lhs}"
    assert relation.rhs in table_names, f"Invalid rhs node type: {relation.rhs}"
    return relation
