import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm_notebook as tqdm
from ultils import load_data_set

CHAR_EMBEDDING_INDEX = {
    " ": 1,
    "!": 2,
    "#": 3,
    "$": 4,
    "%": 5,
    "&": 6,
    "'": 7,
    "(": 8,
    ")": 9,
    "*": 10,
    ",": 11,
    "-": 12,
    ".": 13,
    "/": 14,
    "0": 15,
    "1": 16,
    "2": 17,
    "3": 18,
    "4": 19,
    "5": 20,
    "6": 21,
    "7": 22,
    "8": 23,
    "9": 24,
    ":": 25,
    ";": 26,
    "<": 27,
    ">": 28,
    "?": 29,
    "@": 30,
    "^": 31,
    "a": 32,
    "b": 33,
    "c": 34,
    "d": 35,
    "e": 36,
    "f": 37,
    "g": 38,
    "h": 39,
    "i": 40,
    "j": 41,
    "k": 42,
    "l": 43,
    "m": 44,
    "n": 45,
    "o": 46,
    "p": 47,
    "q": 48,
    "r": 49,
    "s": 50,
    "t": 51,
    "u": 52,
    "v": 53,
    "w": 54,
    "x": 55,
    "y": 56,
    "z": 57,
    "，": 58,
    "<PAD>": 0,
}


def generate_char_embedding(char_to_index, embedding_dim=50):
    """
    Generate embedding matrix
    :param char_to_index: 
    :param embedding_dim: 
    :return: 
    """
    embeddings = np.zeros([len(char_to_index), embedding_dim])
    return torch.from_numpy(np.array(embeddings)).float()


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
    X, X_len = load_padded_data(df, CHAR_EMBEDDING_INDEX, char_level=True)

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
