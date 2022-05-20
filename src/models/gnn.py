import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv


class GAT2017(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_output_heads: int,
        out_dim: int,
        n_layers: int = 2,
        dropout_p: float = 0.6,
    ) -> None:
        super().__init__()
        if n_layers < 2:
            raise ValueError(f"n_layers must be larger or equal to 2: {n_layers=}")

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.ModuleDict(
                {
                    "dropout": nn.Dropout(dropout_p),
                    "conv": GATConv(
                        in_dim, hidden_dim, heads=n_heads, dropout=dropout_p,
                    ),
                    "act": nn.ELU(),
                }
            )
        )
        for _ in range(n_layers - 2):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "dropout": nn.Dropout(dropout_p),
                        "conv": GATConv(
                            hidden_dim * n_heads,
                            hidden_dim,
                            heads=n_heads,
                            dropout=dropout_p,
                        ),
                        "act": nn.ELU(),
                    }
                )
            )
        self.layers.append(
            nn.ModuleDict(
                {
                    "dropout": nn.Dropout(dropout_p),
                    "conv": GATConv(
                        hidden_dim * n_heads,
                        out_dim,
                        heads=n_output_heads,
                        dropout=dropout_p,
                        concat=False,
                    ),
                }
            )
        )

    def forward(self, data):
        x, edge_index = (data.x, data.adj_t)
        for layer in self.layers:
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore
        return x

    def activations(self, data):
        hs = {}
        x, edge_index = (data.x, data.adj_t)
        for i, layer in enumerate(self.layers[:-1], start=1):  # type:ignore
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore
            hs[f"{i}.0"] = x
        return hs


class GCN2017(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        dropout_p: float = 0.6,
    ) -> None:
        super().__init__()
        if n_layers < 2:
            raise ValueError(f"n_layers must be larger or equal to 2: {n_layers=}")

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.ModuleDict(
                {
                    "dropout": nn.Dropout(dropout_p),
                    "conv": GCNConv(in_dim, hidden_dim),
                    "act": nn.ReLU(),
                }
            )
        )
        for _ in range(n_layers - 2):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "dropout": nn.Dropout(dropout_p),
                        "conv": GCNConv(hidden_dim, hidden_dim),
                        "act": nn.ReLU(),
                    }
                )
            )
        self.layers.append(
            nn.ModuleDict(
                {
                    "dropout": nn.Dropout(dropout_p),
                    "conv": GCNConv(hidden_dim, out_dim),
                }
            )
        )

    def forward(self, data):
        x, edge_index = (data.x, data.adj_t)
        for layer in self.layers:
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore
        return x

    def activations(self, data):
        hs = {}
        x, edge_index = (data.x, data.adj_t)
        for i, layer in enumerate(self.layers[:-1], start=1):  # type:ignore
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore
            hs[f"{i}.0"] = x
        return hs

