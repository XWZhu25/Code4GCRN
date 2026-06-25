import torch
from torch import nn


class ChebGraphConv(nn.Module):
    """Polynomial spectral graph convolution corresponding to paper Eq. (7)."""

    def __init__(self, in_dim, out_dim, order):
        super().__init__()
        self.order = order
        self.projection = nn.Linear(in_dim * (order + 1), out_dim)

    def forward(self, x, adjacency):
        supports = [x]
        if self.order >= 1:
            supports.append(torch.einsum("nm,bmf->bnf", adjacency, x))
        for _ in range(2, self.order + 1):
            supports.append(
                2.0 * torch.einsum("nm,bmf->bnf", adjacency, supports[-1])
                - supports[-2]
            )
        return self.projection(torch.cat(supports, dim=-1))


class GCRNCell(nn.Module):
    """GRU gates with dense transforms replaced by graph convolutions."""

    def __init__(self, input_dim, hidden_dim, order):
        super().__init__()
        joint_dim = input_dim + hidden_dim
        self.gates = ChebGraphConv(joint_dim, hidden_dim * 2, order)
        self.candidate = ChebGraphConv(joint_dim, hidden_dim, order)

    def forward(self, x, hidden, adjacency):
        joint = torch.cat([x, hidden], dim=-1)
        reset, update = torch.sigmoid(
            self.gates(joint, adjacency)
        ).chunk(2, dim=-1)
        candidate_input = torch.cat([x, reset * hidden], dim=-1)
        candidate = torch.tanh(self.candidate(candidate_input, adjacency))
        return (1.0 - update) * hidden + update * candidate


class SpatioTemporalAttention(nn.Module):
    """Learns a joint weight for every historical time-node pair."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, sequence):
        scores = self.score(sequence).squeeze(-1)
        weights = torch.softmax(scores.flatten(1), dim=1).view_as(scores)
        context = (sequence * weights.unsqueeze(-1)).sum(dim=1)
        return context, weights


class GCRN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_steps,
        num_layers=2,
        cheb_order=2,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cells = nn.ModuleList(
            [
                GCRNCell(
                    input_dim if layer == 0 else hidden_dim,
                    hidden_dim,
                    cheb_order,
                )
                for layer in range(num_layers)
            ]
        )
        self.attention = SpatioTemporalAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, output_steps)

    def forward(self, x, adjacency, return_attention=False):
        batch, steps, nodes, _ = x.shape
        states = [
            x.new_zeros(batch, nodes, self.hidden_dim) for _ in self.cells
        ]
        encoded_steps = []
        for time_index in range(steps):
            layer_input = x[:, time_index]
            for layer, cell in enumerate(self.cells):
                states[layer] = cell(layer_input, states[layer], adjacency)
                layer_input = self.dropout(states[layer])
            encoded_steps.append(states[-1])

        sequence = torch.stack(encoded_steps, dim=1)
        context, attention = self.attention(sequence)
        prediction = self.output(context).permute(0, 2, 1).unsqueeze(-1)
        return (prediction, attention) if return_attention else prediction
