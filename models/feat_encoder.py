import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatEncoder(nn.Module):
    def __init__(self, topo_dim, hidden_dim, feat_drop):
        super(FeatEncoder, self).__init__()
        self.hidden_dim = hidden_dim

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

        self.feat_trans = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.topo_trans = nn.Linear(topo_dim, hidden_dim, bias=True)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)

    def forward(self, h, ADJ):
        h_pro = F.elu(self.feat_drop(self.feat_trans(h)))
        z_topo = F.elu(self.feat_drop(self.topo_trans(ADJ)))
        z = torch.cat((h_pro, z_topo), 1)
        z_new = F.elu(self.feat_drop(self.fusion(z)))
        return h_pro, z_new
