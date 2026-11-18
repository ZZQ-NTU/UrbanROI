import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.2):
        super().__init__()
        self.gat1 = GATv2Conv(in_dim, hid_dim, heads=4, concat=True,
                              dropout=dropout, edge_dim=1)
        self.gat2 = GATv2Conv(hid_dim * 4, hid_dim, heads=1, concat=False,
                              dropout=dropout, edge_dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        edge_attr = data.edge_attr.reshape(-1, 1)
        x = self.gat1(data.x, data.edge_index, edge_attr=edge_attr)
        x = self.dropout(torch.relu(x))
        x = torch.relu(self.gat2(x, data.edge_index, edge_attr=edge_attr))
        return global_mean_pool(x, data.batch)


class PastROIEncoder(nn.Module):
    def __init__(self, hid_dim, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(1, hid_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, roi_past):
        _, h = self.gru(roi_past.unsqueeze(-1))
        return self.dropout(h[-1])


class AttentionFusion(nn.Module):
    def __init__(self, hid_dim, dropout=0.2, nhead=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hid_dim, nhead, dropout=dropout,
                                          batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.Sequential(nn.LayerNorm(hid_dim), nn.ReLU())

    def forward(self, graph_vec, roi_vec):
        # Attend over both structural and temporal tokens. The original script
        # used one key/value token, yielding a trivial attention distribution.
        query = graph_vec.unsqueeze(1)
        tokens = torch.stack([graph_vec, roi_vec], dim=1)
        attended, _ = self.attn(query, tokens, tokens, need_weights=False)
        return self.norm(graph_vec + self.dropout(attended.squeeze(1)))


class UrbanROI(nn.Module):
    def __init__(self, node_dim, hid_dim=768, future_len=8, dropout=0.2):
        super().__init__()
        self.graph_enc = GraphEncoder(node_dim, hid_dim, dropout)
        self.roi_enc = PastROIEncoder(hid_dim, dropout)
        self.fusion = AttentionFusion(hid_dim, dropout)
        self.baseline_head = nn.Sequential(
            nn.Linear(2 * hid_dim, hid_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid_dim, future_len),
        )
        self.uplift_head = nn.Sequential(
            nn.Linear(3 * hid_dim, hid_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid_dim, future_len),
        )

    def forward(self, g0, g1, roi_past):
        r = self.roi_enc(roi_past)
        z0 = self.fusion(self.graph_enc(g0), r)
        z1 = self.fusion(self.graph_enc(g1), r)
        dz = z1 - z0  # paper Eq. (13); missing in the original single script
        y0 = self.baseline_head(torch.cat([z0, r], dim=-1))
        uplift = self.uplift_head(torch.cat([dz, z0, r], dim=-1))
        y1 = y0 + uplift
        return y0, y1, uplift
