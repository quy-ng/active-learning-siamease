import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.utils import shuffle
import itertools
from multiprocessing.pool import ThreadPool


def load_data_set(file_path):
    df = pd.read_csv(file_path)
    df["content"] = (
        df["address"]
            .str.lower()
            .str.replace("\n", " ")
            .str.replace(r"[ ]+", " ", regex=True)
            .str.replace("null", "")
            .str.replace("nan", "")
    )
    return df


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


def create_data_loader(loader, batch_size):
    array, lengths = np.array(loader["data"]), np.array(loader["length"])
    data = TensorDataset(
        torch.from_numpy(array).type(torch.LongTensor), torch.ByteTensor(lengths)
    )
    return DataLoader(data, batch_size=batch_size, drop_last=False)
