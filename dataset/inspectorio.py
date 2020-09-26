__all__ = ['create_data']

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.utils import shuffle
import itertools
from torch.nn.utils.rnn import pad_sequence
from multiprocessing.pool import ThreadPool
from torch.utils.data import TensorDataset, DataLoader

from ultils.character_level import load_data_set, generate_char_embedding, CHAR_EMBEDDING_INDEX


def create_data_loader(loader, batch_size):
    array, lengths = np.array(loader["data"]), np.array(loader["length"])
    data = TensorDataset(
        torch.from_numpy(array).type(torch.LongTensor), torch.ByteTensor(lengths)
    )
    return DataLoader(data, batch_size=batch_size, drop_last=False)


# todo: write description for prepare_data_for_char func
def prepare_data_for_char(file_path, embedding_dim):
    """
    @param file_path: file to your pre-processed csv
    @param embedding_dim: dim to encode a character (a.k.a dictionary)

    @return: df: Dataframe after processed
    @return: X: full-data as int64 matrix
    @return: X_len: original length of data
    @return: embeddings: character index embeddings index
    """
    # Load dataset
    df = load_data_set(file_path)
    df.fillna("", inplace=True)
    df.reset_index(inplace=True, drop=True)
    # Embedding
    embeddings = generate_char_embedding(CHAR_EMBEDDING_INDEX, embedding_dim)
    # Get X and X_len as matrix
    X, X_len = load_padded_data(df, CHAR_EMBEDDING_INDEX)

    def truncate_non_string(X, X_len):
        # Drop rows that have length of word vector = 0
        truncate_index = [i for i in range(0, len(X_len)) if X_len[i] <= 0]
        X, X_len = (
            np.delete(X, truncate_index, axis=0),
            np.delete(X_len, truncate_index, axis=0),
        )

        return X, X_len, sorted(truncate_index, reverse=True)

    X, X_len, truncate_index = truncate_non_string(X, X_len)
    df.drop(index=truncate_index, inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df, X, X_len, embeddings


def load_padded_data(df, word_to_index):
    """
    Padding data into a fixed length
    :param df: dataframe
    :param word_to_index: dictionary
    :return: x_train_pad: padded data
    :return: x_train_length: original length of all data (in case you want to unpadded)
    """
    x_train = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Padding"):
        words_vector = [word_to_index.get(word) for word in row["content"] if word_to_index.get(word) is not None]
        x_train.append(torch.LongTensor(words_vector))

    x_train_len = [
        len(x) for x in x_train
    ]  # Get length for pack_padded_sequence after to remove padding
    x_train_pad = pad_sequence(x_train, batch_first=True)
    print("Load padded data successfully!")
    return x_train_pad, x_train_len


def load_triplet_orders(df):
    """
    Create triplet samples from dataframe
    :param df:  dataframe
    :return: result with code and content (DataFrame)
    """
    # Generate triplet samples
    cid_list = list(df["cid"].unique())

    def generate_triplet_sample(cid):
        """
        Combine triplet samples
        :param cid:
        :return:
        """
        df_current_pos = df[(df["cid"] == cid) & (df["similar"] == 1)]
        df_current_neg = df[(df["cid"] == cid) & (df["similar"] == 0)]
        # Generate all possible positive
        similar_pairs = list(itertools.combinations(df_current_pos.index, 2))

        if len(similar_pairs) < 2:
            return None

        triplet_order_arr = []
        for each_pair_positive in similar_pairs:
            for neg_sample in df_current_neg.index.values:
                # Link each pair of positive to a negative
                triplet = (cid,) + each_pair_positive + (neg_sample,)
                triplet_order_arr.append(triplet)
        if len(triplet_order_arr) == 0:
            print(cid)
        return np.array(triplet_order_arr)

    # Thread Starting
    pool = ThreadPool()
    threads = [
        pool.apply_async(generate_triplet_sample, (cid,))
        for cid in tqdm(cid_list, desc="[Generate Triplet Dataset] Start thread")
    ]
    # Thread Joining
    result_arr = [
        thread.get()
        for thread in tqdm(threads, desc="[Generate Triplet Dataset] Join thread")
        if thread.get() is not None
    ]
    df_result = pd.DataFrame(np.concatenate(result_arr))
    df_result.rename(columns={0: "cid", 1: "anchor", 2: "pos", 3: "neg"}, inplace=True)
    df_result = shuffle(df_result)
    return df_result


def load_triplet(x_padded, x_lengths, df_triplet_orders, batch_size):
    df_triplet_orders = shuffle(df_triplet_orders)
    anc_dict = {"data": [], "length": []}
    pos_dict = {"data": [], "length": []}
    neg_dict = {"data": [], "length": []}
    for row in tqdm(
            df_triplet_orders.iloc[:, :].itertuples(),
            total=df_triplet_orders.shape[0],
            desc="Load triplets"
    ):
        anc_loc, pos_loc, neg_loc = row[2], row[3], row[4]
        anc_dict["data"].append(x_padded[anc_loc])
        anc_dict["length"].append(x_lengths[anc_loc])

        pos_dict["data"].append(x_padded[pos_loc])
        pos_dict["length"].append(x_lengths[pos_loc])

        neg_dict["data"].append(x_padded[neg_loc])
        neg_dict["length"].append(x_lengths[neg_loc])

    anc_loader = create_data_loader(anc_dict, batch_size)
    pos_loader = create_data_loader(pos_dict, batch_size)
    neg_loader = create_data_loader(neg_dict, batch_size)

    # todo: debug to check return from this func is correct!
    return anc_loader, pos_loader, neg_loader


def create_data(f_path, embeddings_dim, batch_size):
    df, X, X_len, embeddings = prepare_data_for_char(file_path=f_path, embedding_dim=embeddings_dim)
    print('#' * 10)

    df_triplet_orders = load_triplet_orders(df)
    print("Loading triplet order successfully!")

    anc_loader, pos_loader, neg_loader = load_triplet(
        np.array(X), X_len, df_triplet_orders, batch_size=batch_size
    )
    print("Load triplet data successfully!")
    return (anc_loader, pos_loader, neg_loader), (df, X, X_len, embeddings)
