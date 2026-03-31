
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from rdkit import Chem

class GCNModel(nn.Module):
    def __init__(self):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))

        x = torch.mean(x, dim=0)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return torch.sigmoid(x)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    x = torch.ones((num_atoms, 1))

    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    if len(edge_index) == 0:
        return None

    edge_index = torch.tensor(edge_index).t().contiguous()
    return Data(x=x, edge_index=edge_index)

def combine_graphs(smiles1, smiles2, label):
    g1 = smiles_to_graph(smiles1)
    g2 = smiles_to_graph(smiles2)

    if g1 is None or g2 is None:
        return None

    x = torch.cat([g1.x, g2.x], dim=0)
    edge_index = torch.cat([g1.edge_index, g2.edge_index], dim=1)

    return Data(x=x, edge_index=edge_index)
