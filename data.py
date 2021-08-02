import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.decomposition import PCA

from bag_of_words import get_bag_of_words

def get_ironmarch_network_data(posts_path="./data/forum_posts.csv",
                               topics_path="./data/forum_topics.csv",
                               data_path="./data/ironmarch.pth", 
                               threshold=2,
                               load=True,
                               save=True,
                               loadBow=True,
                               saveBow=True,
                               Bow=True):
    '''
    Returns pytorch geometric graph data object created from IronMarch message post data
    and the ids list to look up user database ids

    threshold - determines the number of message topics two users must have posted on together for an edge
    to be created between the two users
    '''

    successful_load = False
    if load and data_path != None:
        try:
            data, ids = torch.load(data_path)
            successful_load = True
        except:
            print(f"load failed for file {data_path} -- creating dataset")

    if not successful_load:
        posts = pd.read_csv(posts_path)
        topics = pd.read_csv(topics_path)

        authors = posts["msg_author_id"].unique()
        mt_ids = topics["mt_id"].unique()

        # Create links (author1 -#-> author2) dictionary
        links = {}
        for mt_id in tqdm(mt_ids, desc="create author links"):
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
        for author in tqdm(links.keys(), desc="create edge index"):
            for co_author in links[author].keys():
                if author != co_author and links[author][co_author] > threshold:
                    edge_index.append([ids.index(author), ids.index(co_author)])

        # for e in tqdm(edge_index, desc="validate edges"):
        #     assert(list(reversed(e)) in edge_index)

        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_index = edge_index.t().contiguous() # this is what pytorch geometric wants

        if Bow:
            # use bag of word features for each user
            bow, vocab, post_authors = get_bag_of_words(posts_path, topics_path, load=loadBow, save=saveBow) # How to place it in so the model will train?
            assert(len(authors) == len(post_authors))

            x = torch.zeros((len(authors), bow.shape[1]))
            for idx, author in enumerate(post_authors):
                x[ids.index(author)] = torch.tensor(bow[idx])

            x = np.array(x)
            scaler = preprocessing.StandardScaler().fit(x)

            x = scaler.transform(x)

            pca = PCA(n_components = 1000)
            pca.fit(x)
            x = pca.transform(x)
            x = torch.tensor(x)
        else:
            # essentially don't use user features
            x = torch.ones((len(authors), 1))

     

        data = Data(x, edge_index=edge_index)

        if save == True and data_path != None:
            torch.save((data, ids), data_path)

    return data, ids

if __name__ == "__main__":
    # get_ironmarch_network_data(load=False, loadBow=False, save=True,
    # #  posts_path="./data/iron_march_201911/csv/core_message_posts.csv", topics_path="./data/iron_march_201911/csv/core_message_topics.csv"
    #  )
    get_ironmarch_network_data(threshold=50, data_path="./data/ironmarch_50.pth", load=False, loadBow=False)
