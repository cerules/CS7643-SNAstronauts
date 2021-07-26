import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import collections, re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords



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
    x = torch.ones((len(authors), 1))

    # TODO: Create bag-of-words
    # Get vocabulary
    # Remove html tags and strip new lines
    # From https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
    TAG_RE = re.compile(r'<[^>]+>')
    def remove_tags(text):
        return TAG_RE.sub('', text).strip().replace('\n', ' ')
    post_list = []
    for post in posts.msg_post.values:
        stripped_post = remove_tags(post)
        post_list.append(stripped_post)
    # Get unique words and remove common words
    # Source: https://stackoverflow.com/questions/46360435/how-to-create-a-bag-of-words-from-a-pandas-dataframe
    msg_post_word_counter = collections.Counter([y for x in post_list for y in x.split()])
    filtered_post_words = [word for word in msg_post_word_counter if word not in stopwords.words('english')]
    # TODO: Create word-to-idx vector
    # Source: https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
    # word_to_ix = {"hello": 0, "world": 1}
    # embeds = torch.nn.Embedding(len(filtered_post_words), 5)  # n words in vocab, 5 dimensional embeddings
    # print(embeds)
    # TODO: Need to actually implement. Also split this into a separate module
    # Sources: https://mmuratarat.github.io/2020-04-03/bow_model_tf_idf, ...

    data = Data(x=x, edge_index=edge_index)

    return data, ids

if __name__ == "__main__":
    get_ironmarch_network_data()