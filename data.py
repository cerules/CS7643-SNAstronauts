import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data


def get_ironmarch_network_data(posts_path = "./data/iron_march_201911/csv/core_message_posts.csv", 
                               topics_path = "./data/iron_march_201911/csv/core_message_topics.csv", 
                               threshold = 2):
    '''
    Returns pytorch geometric graph data object created from IronMarch message post data
    and the ids list to look up user database ids

    threshold - determines the number of message topics two users must have posted on together for an edge 
    to be created between the two users 
    '''
    posts = pd.read_csv(posts_path)
    topics = pd.read_csv(topics_path)

    authors = posts['msg_author_id'].unique()
    mt_ids = topics["mt_id"].unique()

    # Create links (author1 -#-> author2) dictionary 
    links = {}
    for mt_id in mt_ids:
        mt_posts = posts.loc[posts['msg_topic_id'] == mt_id]
        mt_post_authors = mt_posts["msg_author_id"].unique()
        for author in mt_post_authors:
            # Creating links
            if author not in links:
                links[author] = {}
            for other_author in mt_post_authors:
                if other_author not in links[author]:
                    links[author][other_author] = 0
                links[author][other_author] = links[author][other_author] + 1

    # Need ids list to map edge index to member id
    ids = list(links.keys())
    ids.sort()

    # Create edge_index list of [author1, author2] pairs that meet the threshold
    edge_index = []
    for author in links.keys():
        for co_author in links[author].keys():
            if author != co_author and links[author][co_author] > threshold:
                edge_index.append([ids.index(author), ids.index(co_author)])

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.t().contiguous() # this is what pytorch geometric wants

    # just going to give each user a feature vector of 1 until we get bag of words
    # x = torch.ones((len(authors), 1))
    x = torch.tensor((len(authors), ))

    data = Data(x=x, edge_index=edge_index)

    return data, ids

if __name__ == "__main__":
    get_ironmarch_network_data()
