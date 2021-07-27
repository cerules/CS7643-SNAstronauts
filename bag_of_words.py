import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import collections, re

# Alternatives:
from keras.preprocessing.text import Tokenizer
# import torchtext
# from torchtext.data import get_tokenizer


def get_bag_of_words():
    '''
    Returns bag of words tensor for each other given the post vocabulary

    TODO: Main issue is the memory issue still. Efficient stop word filtering without looking
    at every single word. Keras is the best I've found for getting the desired representation.
    Feel free to play around with this.
    '''
    posts = pd.read_csv("./data/iron_march_201911/csv/core_message_posts.csv")
    topics = pd.read_csv("./data/iron_march_201911/csv/core_message_topics.csv")

    authors = posts['msg_author_id'].unique()
    mt_ids = topics["mt_id"].unique()

    # Create bag-of-words
    # ---------- Get vocabulary (create lists of posts) ----------
    # Remove html tags and strip new lines
    # # From https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
    # TAG_RE = re.compile(r'<[^>]+>')
    # def remove_tags(text):
    #     return TAG_RE.sub('', text).strip().replace('\n', ' ')
    post_list = []
    for post in posts.msg_post.values:
        stripped_post = BeautifulSoup(post, 'html.parser').get_text().replace(u'\xa0', u' ')
        post_list.append(stripped_post)
    print(f"len(post_list) = {len(post_list)}")

    # # TODO: Get unique words and remove common words. Memory issue remains though
    # # Source: https://stackoverflow.com/questions/46360435/how-to-create-a-bag-of-words-from-a-pandas-dataframe
    # msg_post_word_counter = collections.Counter([y for x in post_list for y in x.split()])
    # filtered_post_words = [word for word in msg_post_word_counter if word not in stopwords.words('english')]

    # Using keras Tokenizer to create vocabulary
    bow_vocab_model = Tokenizer()
    bow_vocab_model.fit_on_texts(post_list)

    # Print keys 
    # print(f'Key : {list(bow_vocab_model.word_index.keys())}')
    print(f'len(list(bow_vocab_model.word_index.keys())) = {len(list(bow_vocab_model.word_index.keys()))}')

    # ---------- Get bag of words representation for each author based on words in each post ----------
    # Create links (author1 -#-> author2) dictionary
    author_post_bags_of_words = []
    for author in [0, 1]:
        mt_posts = posts.loc[posts['msg_author_id'] == author]
        posts_for_this_author = mt_posts.msg_post.values
        print(type(posts_for_this_author))
        print(len(posts_for_this_author))
        # print(posts_for_this_author)
        # TODO: Need to filter stop words and formatting here as well
        author_bow = bow_vocab_model.texts_to_matrix(posts_for_this_author, mode='count')
        author_post_bags_of_words.append(author_bow)

    # # Alternative: torchtext get_tokenizer
    # tokenizer = get_tokenizer("basic_english")
    # tokens = tokenizer("You can now install TorchText using pip!")
    # print(tokens)

    return torch.tensor(author_post_bags_of_words), bow_vocab_model

if __name__ == "__main__":
    get_bag_of_words()
