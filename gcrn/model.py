import torch
from torch import nn


class ChebGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, order):
        super().__init__()
        self.order = order
        self.projection = nn.Linear(in_dim * (order + 1), out_dim)

    def forward(self, x, adj):
        supports = [x]
        if self.order >= 1:
            supports.append(torch.einsum("nm,bmf->bnf", adj, x))

        for _ in range(2, self.order + 1):
            next_support = 2 * torch.einsum(
                "nm,bmf->bnf", adj, supports[-1]
            ) - supports[-2]
            supports.append(next_support)

        return self.projection(torch.cat(supports, dim=-1))


class GCRNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, order):
        super().__init__()
        joint_dim = input_dim + hidden_dim
        self.gates = ChebGraphConv(joint_dim, hidden_dim * 2, order)
        self.candidate = ChebGraphConv(joint_dim, hidden_dim, order)

    def forward(self, x, hidden, adj):
        joint = torch.cat([x, hidden], dim=-1)
        reset, update = torch.sigmoid(self.gates(joint, adj)).chunk(2, dim=-1)
        candidate_input = torch.cat([x, reset * hidden], dim=-1)
        candidate = torch.tanh(self.candidate(candidate_input, adj))
        return (1 - update) * hidden + update * candidate


class SpatioTemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, sequence):
        scores = self.score(sequence).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
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

    def forward(self, x, adj, return_attention=False):
        batch_size, num_steps, num_nodes, _ = x.shape
        states = [
            x.new_zeros(batch_size, num_nodes, self.hidden_dim)
            for _ in self.cells
        ]
        history = []
        for t in range(num_steps):
            current = x[:, t]
            for i, cell in enumerate(self.cells):
                states[i] = cell(current, states[i], adj)
                current = self.dropout(states[i])
            history.append(states[-1])

        sequence = torch.stack(history, dim=1)
        context, attention = self.attention(sequence)
        prediction = self.output(context).permute(0, 2, 1).unsqueeze(-1)
        if return_attention:
            return prediction, attention
        return prediction
