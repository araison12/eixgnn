Here is the an example code for using EiXGNN from the [EiX-GNN: Concept-level eigencentrality explainer for graph neural
    networks](https://arxiv.org/abs/2206.03491) paper

```python
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root="/tmp/ENZYMES", name="ENZYMES")
data = dataset[0]

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from eixgnn.eixgnn import EiXGNN

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 20)
        self.conv2 = GCNConv(20, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.softmax(x, dim=1)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN().to(device)
model.eval()
data = dataset[0].to(device)
explainer = EiXGNN()
explained = explainer.forward(model, data.x, data.edge_index)
```
