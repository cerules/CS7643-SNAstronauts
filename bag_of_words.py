import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

from nltk import word_tokenize
# nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from bs4 import BeautifulSoup
import collections, re

from keras.preprocessing.text import Tokenizer
# import torchtext
# from torchtext.data import get_tokenizer


def get_bag_of_words(posts_path = "./data/iron_march_201911/csv/core_message_posts.csv", 
                     topics_path = "./data/iron_march_201911/csv/core_message_topics.csv"):
    '''
    Creates a post vocabulary

    return 
    0 - Bag of words array of shape (number_of_authors, number_of_words_in_vocab) 
        where the vocabulary is pulled from all the posts. The number frequency of the vocab word that
        shows up in the authors' posts is at each number_of_words_in_vocab index.
    1 - List holding the vocabulary from this particular complete dataset.
    '''
    posts = pd.read_csv(posts_path)
    topics = pd.read_csv(topics_path)

    authors = posts['msg_author_id'].unique()
    mt_ids = topics["mt_id"].unique()

    # Create bag-of-words
    # ---------- Get vocabulary (create lists of posts) ----------
    # Remove html tags, stopwords, and punctuation
    post_list = []
    # print(posts.msg_post.values.shape) # debugging print
    for post in posts.msg_post.values:
        # print(f'post = {post}') # debugging print
        stripped_post = BeautifulSoup(post, 'html.parser').get_text().replace(u'\xa0', u' ')
        # Remove stopwords (must look at every word so slows performance)
        stop = stopwords.words('english')
        stripped_post = ' '.join([i for i in word_tokenize(stripped_post.lower()) if i not in stop])
        post_list.append(stripped_post)
        # print(f"post_list = {post_list}") # debugging print
        # print(f"len(post_list) = {len(post_list)}") # debugging print

    # Using keras Tokenizer to create vocabulary
    bow_vocab_model = Tokenizer()
    bow_vocab_model.fit_on_texts(post_list)

    # Print keys # debugging print
    # print(f'Key : {list(bow_vocab_model.word_index.keys())}')
    print(f'len(list(bow_vocab_model.word_index.keys())) = {len(list(bow_vocab_model.word_index.keys()))}')

    # ---------- Get bag of words representation for each author based on words in each post ----------
    author_post_bags_of_words = []
    for author in authors:
        mt_posts = posts.loc[posts['msg_author_id'] == author]
        posts_for_this_author = mt_posts.msg_post.values
        # print(f'type(posts_for_this_author) = {type(posts_for_this_author)}') # debugging print
        # print(f'posts_for_this_author.shape = {posts_for_this_author.shape}') # debugging print
        # print(posts_for_this_author) # debugging print
        posts_for_this_author_list = []
        # Remove text similar to vocab
        for post in posts_for_this_author:
            stripped_post = BeautifulSoup(post, 'html.parser').get_text().replace(u'\xa0', u' ')
            posts_for_this_author_list.append(stripped_post)
        author_bow = bow_vocab_model.texts_to_matrix(posts_for_this_author_list, mode='count')
        # print(f'type(author_bow) = {type(author_bow)}') # debugging print
        # print(f'author_bow.shape = {author_bow.shape}') # debugging print
        author_bow = np.sum(author_bow, axis=0)
        # print("Summing along axis=0") # debugging print
        # print(f'author_bow = {author_bow}') # debugging print
        # print(f'author_bow.shape = {author_bow.shape}') # debugging print
        author_post_bags_of_words.append(author_bow)

    # There's an extra column for some reason. Remove here but should find the cause later
    # print(f'np.array(author_post_bags_of_words)[:, 1:] = {np.array(author_post_bags_of_words)[:, 1:]}') # debugging print
    print(f'np.array(author_post_bags_of_words)[:, 1:].shape = {np.array(author_post_bags_of_words)[:, 1:].shape}') # debugging print

    return np.array(author_post_bags_of_words)[:, 1:], list(bow_vocab_model.word_index.keys())

if __name__ == "__main__":
    get_bag_of_words()
