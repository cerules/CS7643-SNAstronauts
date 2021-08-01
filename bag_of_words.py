import pandas as pd
import numpy as np
import math
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from nltk import word_tokenize
# nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from bs4 import BeautifulSoup
import collections, re

from keras.preprocessing.text import Tokenizer
# import torchtext
# from torchtext.data import get_tokenizer


def get_bag_of_words(posts_path = "./data/forum_posts.csv",
                        topics_path = "./data/forum_topics.csv", save_path="./data/bow.npy", load=True, save=True):
    '''
    Creates a post vocabulary

    return 
    0 - Bag of words array of shape (number_of_authors, number_of_words_in_vocab) 
        where the vocabulary is pulled from all the posts. The number frequency of the vocab word that
        shows up in the authors' posts is at each number_of_words_in_vocab index.
    1 - List holding the vocabulary from this particular complete dataset.
    '''
    successful_load = False
    if load and save_path != None:
        try:
            bow, vocab, authors = np.load(save_path, allow_pickle=True)
            successful_load = True
        except:
            print(f"load failed for file {save_path} -- recreating BOW")
    if not successful_load:
        posts = pd.read_csv(posts_path)
        topics = pd.read_csv(topics_path)

        authors = posts['msg_author_id'].unique()
        mt_ids = topics["mt_id"].unique()

        # Create bag-of-words
        # ---------- Get vocabulary (create lists of posts) ----------
        # Remove html tags, stopwords, and punctuation
        post_list = []
        post_token_list = []
        # print(posts.msg_post.values.shape) # debugging print
        for post in tqdm(posts.msg_post.values, desc="create vocabulary"):
            # print(f'post = {post}') # debugging print
           
            if post is not None and type(post) == str:
                stripped_post = BeautifulSoup(post, 'html.parser').get_text().replace(u'\xa0', u' ')
                # Remove stopwords (must look at every word so slows performance)
                stop = stopwords.words('english')
                tokens = [i for i in word_tokenize(stripped_post.lower()) if i not in stop and i.isalpha()]
                post_token_list.append(tokens)
                stripped_post = ' '.join(tokens)
                post_list.append(stripped_post)
                # print(f"post_list = {post_list}") # debugging print
                # print(f"len(post_list) = {len(post_list)}") # debugging print
            else:
                post_list.append("")

        # Using keras Tokenizer to create vocabulary
        bow_vocab_model = Tokenizer(num_words=50000)
        bow_vocab_model.fit_on_texts(post_token_list)
        posts['stripped_post'] = post_list

        # Print keys # debugging print
        # print(f'Key : {list(bow_vocab_model.word_index.keys())}')
        print(f'len(list(bow_vocab_model.word_index.keys())) = {len(list(bow_vocab_model.word_index.keys()))}')

        # ---------- Get bag of words representation for each author based on words in each post ----------
        author_post_bags_of_words = []
        for author in tqdm(authors, desc="processing authors"):
            mt_posts = posts.loc[posts['msg_author_id'] == author]
            posts_for_this_author = mt_posts.stripped_post.values
            # print(f'type(posts_for_this_author) = {type(posts_for_this_author)}') # debugging print
            # print(f'posts_for_this_author.shape = {posts_for_this_author.shape}') # debugging print
            # print(posts_for_this_author) # debugging print
            posts_for_this_author_list = []
            # Remove text similar to vocab
            for stripped_post in posts_for_this_author:
                if stripped_post is not None and type(post) == str:
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

        bow = np.array(author_post_bags_of_words)[:, 1:]
        vocab = list(bow_vocab_model.word_index.keys())

        if save and save_path != None:
            np.save(save_path, (bow, vocab, authors))
        

    return bow, vocab, authors

if __name__ == "__main__":
    get_bag_of_words()
