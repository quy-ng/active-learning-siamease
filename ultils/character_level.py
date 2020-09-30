import numpy as np
import torch
import pandas as pd

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


def generate_char_embedding(char_to_index=CHAR_EMBEDDING_INDEX, embedding_dim=50):
    """
    Generate embedding matrix
    :param char_to_index: 
    :param embedding_dim: 
    :return: 
    """
    embeddings = np.zeros([len(char_to_index), embedding_dim])
    return torch.from_numpy(np.array(embeddings)).float()
