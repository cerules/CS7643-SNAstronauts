# from Pytorch Geometric link prediction example
# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/link_pred.py

from data import get_ironmarch_network_data
from gnn import Net
from train import Trainer
import torch
from torch_geometric.utils import train_test_split_edges
import numpy as np
import random

if __name__ == "__main__":

    # set seeds
    seed = 27
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    data, _ = get_ironmarch_network_data(threshold=3, load=True, loadBow=True, data_path="./data/report_ironmarch_3_bow_pca_1000.pth")
    data = train_test_split_edges(data)
    num_features = data.x.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(num_features, 64).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, verbose=True)

    trainer = Trainer(model, optimizer, device, "report_ironmarch_3_bow_pca_1000", scheduler=scheduler)

    trainer.train(data, 1001)

    z = model.encode(data.x, data.train_pos_edge_index)
    final_edge_index = model.decode_all(z)
