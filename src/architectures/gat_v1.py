import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm
from torch_geometric.nn import TopKPooling
import torch.nn as nn
from torch.nn import Linear
import torch

"""
Applying pyG lib
"""


class GATv1(nn.Module):
    def __init__(
            self,
            node_feature_dim,
            hidden_layer_dimension,
            numbers_of_class,
            dropout_rate,
            number_of_heads,
            pooling_ratio,
            add_self_loops,
            device,
    ):
        super(GATv1, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_layer_dimension = hidden_layer_dimension
        self.numbers_of_class = numbers_of_class
        self.dropout_rate = dropout_rate
        self.number_of_heads = number_of_heads
        self.pooling_ratio = pooling_ratio
        self.add_self_loops = add_self_loops
        self.device = device

        self.conv1 = GATConv(
            in_channels=self.node_feature_dim,
            out_channels=self.hidden_layer_dimension,
            heads=self.number_of_heads,
            add_self_loops=self.add_self_loops
        )

        self.conv2 = GATConv(
            in_channels=self.number_of_heads * self.hidden_layer_dimension,
            out_channels=self.hidden_layer_dimension,
            heads=self.number_of_heads,
            add_self_loops=self.add_self_loops
        )

        self.conv3 = GATConv(
            in_channels=self.number_of_heads * self.hidden_layer_dimension,
            out_channels=self.hidden_layer_dimension,
            heads=self.number_of_heads,
            concat=False,
            add_self_loops=self.add_self_loops
        )

        self.norm1 = LayerNorm(self.number_of_heads * self.hidden_layer_dimension)
        self.norm2 = LayerNorm(self.number_of_heads * self.hidden_layer_dimension)
        self.norm3 = LayerNorm(self.hidden_layer_dimension)

        self.lin0 = Linear(self.hidden_layer_dimension, self.hidden_layer_dimension)
        self.lin1 = Linear(self.hidden_layer_dimension, self.hidden_layer_dimension)
        self.lin = Linear(self.hidden_layer_dimension, self.numbers_of_class)

        self.topk_pool = TopKPooling(self.hidden_layer_dimension, ratio=self.pooling_ratio)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self,
            x,
            edge_index,
            edge_attr,
            batch
    ):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x, batch)

        x = self.topk_pool(x, edge_index, edge_attr, batch=batch)[0]
        x = torch.transpose(x, 0, 1)
        x = nn.Linear(x.shape[1], batch[-1] + 1, bias=False, device=self.device)(x)
        x = torch.transpose(x, 0, 1)

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.lin0(x)
        x = F.relu(x)

        x = self.lin1(x)
        x = F.relu(x)

        z = x  # extract last layer features

        x = self.lin(x)

        return x, z


