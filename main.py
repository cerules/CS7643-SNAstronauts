
# from Pytorch Geometric link prediction example
# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/link_pred.py

from data import get_ironmarch_network_data
from gnn import Net, train, test
import torch
from torch_geometric.utils import train_test_split_edges

if __name__ == "__main__":
    data, _ = get_ironmarch_network_data()
    data = train_test_split_edges(data)
    num_features = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_features, 64).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    best_val_auc = test_auc = 0
    for epoch in range(1, 10001):
        loss = train(model, optimizer, device, data)
        val_auc, tmp_test_auc = test(model, device, data)
        if val_auc > best_val_auc:
            best_val = val_auc
            test_auc = tmp_test_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
            f'Test: {test_auc:.4f}')

    z = model.encode(data.x, data.train_pos_edge_index)
    final_edge_index = model.decode_all(z)
