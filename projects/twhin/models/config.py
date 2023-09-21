import typing
import enum

from tml.common.modules.embedding.config import LargeEmbeddingsConfig
from tml.core.config import base_config
from tml.optimizers.config import OptimizerConfig

import pydantic
from pydantic import validator


class TwhinEmbeddingsConfig(LargeEmbeddingsConfig):
  """
    Configuration class for Twhin model embeddings.

    This class inherits from LargeEmbeddingsConfig and ensures that the embedding dimensions and data types
    for all tables in the Twhin model embeddings configuration match.

    Attributes:
        tables (List[TableConfig]): A list of table configurations for the model's embeddings.
    """
  @validator("tables")
  def embedding_dims_match(cls, tables):
    """
        Validate that embedding dimensions and data types match for all tables.

        Args:
            tables (List[TableConfig]): List of table configurations.

        Returns:
            List[TableConfig]: The list of validated table configurations.

        Raises:
            AssertionError: If embedding dimensions or data types do not match.
        """
    embedding_dim = tables[0].embedding_dim
    data_type = tables[0].data_type
    for table in tables:
      assert table.embedding_dim == embedding_dim, "Embedding dimensions for all nodes must match."
      assert table.data_type == data_type, "Data types for all nodes must match."
    return tables


class Operator(str, enum.Enum):
  """
    Enumeration of operator types.

    This enumeration defines different types of operators that can be applied to Twhin model relations.
    """
  TRANSLATION = "translation"


class Relation(pydantic.BaseModel):
  """
    Configuration class for graph relationships in the Twhin model.

    This class defines properties and operators for graph relationships in the Twhin model.

    Attributes:
        name (str): The name of the relationship.
        lhs (str): The name of the entity on the left-hand side of the relation.
        rhs (str): The name of the entity on the right-hand side of the relation.
        operator (Operator): The transformation operator to apply to the left-hand side embedding before dot product.
    """

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
  """
    Configuration class for the Twhin model.

    This class defines configuration options specific to the Twhin model.

    Attributes:
        embeddings (TwhinEmbeddingsConfig): Configuration for the model's embeddings.
        relations (List[Relation]): List of graph relationship configurations.
        translation_optimizer (OptimizerConfig): Configuration for the optimizer used for translation.
    """
  embeddings: TwhinEmbeddingsConfig
  relations: typing.List[Relation]
  translation_optimizer: OptimizerConfig

  @validator("relations", each_item=True)
  def valid_node_types(cls, relation, values, **kwargs):
    """
        Validate that the specified node types in relations are valid table names in embeddings.

        Args:
            relation (Relation): A single relation configuration.
            values (dict): The values dictionary containing the "embeddings" configuration.

        Returns:
            Relation: The validated relation configuration.

        Raises:
            AssertionError: If the specified node types are not valid table names in embeddings.
        """
    table_names = [table.name for table in values["embeddings"].tables]
    assert relation.lhs in table_names, f"Invalid lhs node type: {relation.lhs}"
    assert relation.rhs in table_names, f"Invalid rhs node type: {relation.rhs}"
    return relation
