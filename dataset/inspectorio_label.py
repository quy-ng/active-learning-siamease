import itertools
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.utils import shuffle
from multiprocessing.pool import ThreadPool
from ultils.character_level import vectorize
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

__all__ = ['InspectorioLabel']


def load_data_set(
        file_path="generated_labeled_data.csv"
):
    """
    Loads the dataset with self-attraction embedding
    :param file_path: main file
    :param retrain: if you want to retrain model
    :return: dataframe
    """

    df = pd.read_csv(file_path)
    df["content"] = (
        df["address"]
            .str.lower()
            .str.replace("\n", " ")
            .str.replace(r"[ ]+", " ", regex=True)
            .str.replace("null", "")
            .str.replace("nan", "")
    )
    df.fillna("", inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def load_padded_data(
        df,
        word_to_index
):
    """
    Padding data into a fixed length
    :param df: dataframe
    :param word_to_index: dictionary
    :param retrain: (default is False) True if you want to retrain
    :return: x_train_pad: padded data
    :return: x_train_length: original length of all data (in case you want to unpadded)
    """
    x_train = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Padding"):
        # Create word vector
        chars_vector = vectorize(row["content"], word_to_index)
        x_train.append(torch.LongTensor(chars_vector))

    x_train_len = [
        len(x) for x in x_train
    ]  # Get length for pack_padded_sequence after to remove padding
    x_train_pad = pad_sequence(x_train, batch_first=True)
    print("Load padded data successfully!")
    return x_train_pad, x_train_len


def prepare_data(file_path, vocab):
    # Load dataset
    df = load_data_set(file_path)
    df.fillna("", inplace=True)
    df.reset_index(inplace=True, drop=True)

    # Get X and X_len as matrix
    X, X_len = load_padded_data(df, vocab)

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

    return df, X, X_len


def load_triplet_orders(df):
    """
    Create triplet samples
    :param df:  dataframe
    :param retrain:  True if you want to re-generate triplet order (Default False)
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

    # Retrain FALSE

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


def load_triplet(
        x_padded,
        x_lengths,
        df_triplet_orders,
        batch_size=520
):
    df_triplet_orders = shuffle(df_triplet_orders)
    anc_dict = {"data": [], "length": []}
    pos_dict = {"data": [], "length": []}
    neg_dict = {"data": [], "length": []}
    for row in tqdm(
            df_triplet_orders.iloc[:, :].itertuples(),
            total=df_triplet_orders.shape[0],
            desc="Load triplets",
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

    return anc_loader, pos_loader, neg_loader


def create_data_loader(loader, batch_size=5000):
    array, lengths = np.array(loader["data"]), np.array(loader["length"])
    data = TensorDataset(
        torch.from_numpy(array).type(torch.LongTensor), torch.ByteTensor(lengths)
    )
    return DataLoader(data, batch_size=batch_size, drop_last=False)


class InspectorioLabel:
    @classmethod
    def load_data(cls, data_path, batch_size, vocab):
        # ---- Load data and convert it to triplet
        df, X, X_len = prepare_data(data_path, vocab)
        # Create data loader with batch
        df_triplet_orders = load_triplet_orders(df)
        print("Loading triplet order successfully!")
        anc_loader, pos_loader, neg_loader = load_triplet(
            np.array(X), X_len, df_triplet_orders, batch_size=batch_size
        )
        print("Load triplet data successfully!")
        return anc_loader, pos_loader, neg_loader, max(X_len)


if __name__ == '__main__':
    from ultils.character_level import default_vocab

    anc_loader, pos_loader, neg_loader = InspectorioLabel.load_data(
        '../data/dac/dedupe-project/new/new_generated_labeled_data.csv', 2, default_vocab
    )

    for batch, [anc_x, pos_x, neg_x] in enumerate(
            zip(anc_loader, pos_loader, neg_loader)
    ):
        # Send data to graphic card - Cuda
        print(anc_x, pos_x, neg_x)
        break
