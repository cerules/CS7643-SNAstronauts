import networkx as nx
import matplotlib.pyplot as plt
from data import get_ironmarch_network_data
import torch
from gnn import Net
import numpy as np

def inference():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, _ = get_ironmarch_network_data("./data/forum_posts.csv", "./data/forum_topics.csv", data_path="./data/ironmarch_5.pth")
    data = data.to(device)
    num_features = data.x.shape[1]
    model = Net(num_features, 64).to(device)
    model.load_state_dict(torch.load("./output/1000_epochs_ironmarch_5/best.pth"))
    model.eval()

    z = model.encode(data.x, data.edge_index)
    inferred_edge_index = model.decode_all(z)
    draw_graph(inferred_edge_index)
    print(inferred_edge_index)

def draw_graph(edge_index, filename="test_el.csv"):
    G = nx.Graph()
    for edge in range(edge_index.shape[1]):
        G.add_edge(edge_index[0][edge].item(), edge_index[1][edge].item())
    nx.write_edgelist(G, filename, delimiter=',')
    # fig, ax = plt.subplots()
    #
    # nx.draw(G)
    #
    # plt.show()
# def draw_graph(edge_index):
#     G = nx.Graph()

#     for edge in range(edge_index.shape[1]):
#         G.add_edge(edge_index[0][edge], edge_index[1][edge])

#     fig, ax = plt.subplots()

#     nx.draw(G)

#     plt.show()


if __name__ == "__main__":
    # data, _ = get_ironmarch_network_data("./data/forum_posts.csv", "./data/forum_topics.csv")
    # draw_graph(data.edge_index)
    inference()
