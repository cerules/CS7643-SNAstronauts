import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data



def get_ironmarch_network_data(posts_path = "./data/iron_march_201911/csv/core_message_posts.csv", topics_path = "./data/iron_march_201911/csv/core_message_topics.csv", threshold = 2):
    '''
    Returns pytorch geometric graph data object created from IronMarch message post data

    threshold determines the number of message topics two users must have posted on together for an edge to be created between the two users 
    '''
    posts = pd.read_csv(posts_path)
    topics = pd.read_csv(topics_path)

    authors = posts['msg_author_id'].unique()
    mt_ids = topics["mt_id"].unique()

    links = {}
    for mt_id in mt_ids:
        mt_posts = posts.loc[posts['msg_topic_id'] == mt_id]
        mt_post_authors = mt_posts["msg_author_id"].unique()
        for author in mt_post_authors:
            if author not in links:
                links[author] = {}
            for other_author in mt_post_authors:
                if other_author not in links[author]:
                    links[author][other_author] = 0
                links[author][other_author] = links[author][other_author] + 1

    edge_index = []
    for author in links.keys():
        for co_author in links[author].keys():
            if author != co_author and links[author][co_author] > threshold:
                edge_index.append([author, co_author])

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index=edge_index.t().contiguous() # this is what pytorch geometric wants

    # just going to give each user a feature vector of 0 until we get bag of words
    x = torch.zeros(len(authors))

    data = Data(x=x, edge_index=edge_index)

    return data