from typing import Optional
from pydantic.v1 import BaseModel, PositiveFloat, PositiveInt

import torch
from torch import nn
from src.architecture.gat_v1 import GATv1
from src.architecture.gat_v2 import GATv2

from src.config.types import GDLArchitecture


class GNNParametersDTO(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    node_feature_dim: Optional[PositiveInt]
    hidden_layer_dimension: PositiveInt
    numbers_of_class: PositiveInt
    dropout_rate: PositiveFloat
    number_of_heads: PositiveInt
    pooling_ratio: PositiveInt
    add_self_loops: bool
    device: torch.device


class GNNFactory:
    def __init__(self, gdl_architecture: GDLArchitecture):
        self._gdl_architecture = gdl_architecture

    def create(self, gnn_parameters_dto: GNNParametersDTO) -> nn.Module:
        if self._gdl_architecture == GDLArchitecture.GATV1:
            return GATv1(**gnn_parameters_dto.dict())
        if self._gdl_architecture == GDLArchitecture.GATV2:
            return GATv2(**gnn_parameters_dto.dict())
        else:
            raise ValueError(f"Unsupported GNN type: {self._gdl_architecture}")