import torch.nn as nn
from torch_geometric.nn import GraphSAGE, global_mean_pool

class GraphGNNModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=4):
        super(GraphGNNModel, self).__init__()

        self.sages = nn.ModuleList([
            GraphSAGE(c_in, c_hidden, num_layers= num_layers),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),

            GraphSAGE(c_hidden, c_hidden, num_layers= num_layers),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),

            GraphSAGE(c_hidden, c_hidden, num_layers= num_layers)
        ])

        self.head = nn.Linear(c_hidden, c_out)

    def forward(self, x, edge_index, batch_idx):
        for sage in self.sages:
            if isinstance(sage, GraphSAGE):
                x = sage(x, edge_index)
            else:
                x = sage(x)

        x = global_mean_pool(x, batch_idx) # Average pooling
        x = self.head(x)
        return x
