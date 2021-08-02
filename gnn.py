# from Pytorch Geometric link prediction example
# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/link_pred.py
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score

from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges, accuracy

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        
        

        # hidden=2048

        # self.conv1 = GCNConv(in_channels, hidden)
        # self.conv2 = GCNConv(hidden, hidden // 2)
        # self.conv3 = GCNConv(hidden // 2, hidden // 4)
        # self.conv4 = GCNConv(hidden // 4, hidden // 8)
        # self.conv5 = GCNConv(hidden // 8, hidden // 16)
        # self.conv6 = GCNConv(hidden // 16, hidden // 32)

        # self.conv_out = GCNConv(hidden // 32, out_channels)

        
        # self.convs = [
        #     self.conv1,
        #     self.conv2,
        #     self.conv3,
        #     self.conv4,
        #     self.conv5,
        #     self.conv6
        # ]
        self.conv1 = GCNConv(in_channels, 128)
        self.conv3 = GCNConv(128, 1024)
        self.conv4 = GCNConv(1024, 1024)
        self.conv5 = GCNConv(1024, 1024)
        self.conv6 = GCNConv(1024, 1024)
        self.conv2 = GCNConv(1024, out_channels)

    def encode(self, x, edge_index):
        # for conv in self.convs:
        #     x = conv(x, edge_index)
        #     x = x.relu()
  
        # return self.conv_out(x, edge_index)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = x.relu()
        x = self.conv5(x, edge_index)
        x = x.relu()
        x = self.conv6(x, edge_index)
        x = x.relu()
        return self.conv2(x, edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

def get_link_labels(pos_edge_index, neg_edge_index, device):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def train_epoch(model, optimizer, device, data):
    model.train()

    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))

    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index, device)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    link_probs = link_logits.sigmoid()
    train_auc = roc_auc_score(link_labels.cpu().detach().numpy(), link_probs.cpu().detach().numpy())
    train_acc = accuracy(link_labels, (link_probs > .5))

    return loss, train_auc, train_acc

@torch.no_grad()
def test(model, device, data):
    model.eval()

    z = model.encode(data.x, data.train_pos_edge_index)

    results = []
    for prefix in ['val', 'test']:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index, device)
        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        results.append(loss)
        results.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
        results.append(accuracy(link_labels, (link_probs > .5)))
    return results
