
# from Pytorch Geometric link prediction example
# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/link_pred.py

from data import get_ironmarch_network_data
from gnn import Net
from train import Trainer
import torch
from torch_geometric.utils import train_test_split_edges

if __name__ == "__main__":
    data, _ = get_ironmarch_network_data(threshold=5, load=True, data_path="./data/ironmarch_5.pth")
    data = train_test_split_edges(data)
    num_features = data.x.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_features, 64).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    trainer = Trainer(model, optimizer, device, "1000_epochs_ironmarch_5")

    trainer.train(data, 1001)

    z = model.encode(data.x, data.train_pos_edge_index)
    final_edge_index = model.decode_all(z)
