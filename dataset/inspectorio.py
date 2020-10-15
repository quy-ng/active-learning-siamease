import re
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader
from dataset.inspectorio_label import load_padded_data


__all__ = ['Inspectorio']


def remove_alone_char(text):
    regex = r'\s(\-|\.|\,)\s'
    matches = re.finditer(regex, text, re.MULTILINE)
    flag = False
    for matchNum, match in enumerate(matches, start=1):
        a = match.start()
        b = match.end()
        sub_text = text[a:b]
        sub_text = re.sub(r'(-|.|,)', ' ', sub_text)
        text = ' '.join([text[0:a], sub_text, text[b:]])
        flag = True
        break
    if flag:
        return remove_alone_char(text)
    return text


def process_helper(row):
    result_address = row
    result_address = result_address.lower()
    result_address = re.sub('null', ' ', result_address)
    result_address = re.sub(r'(0)\1{4}', ' ', result_address)  # remove zipcode 00000
    result_address = re.sub(r'(,)\1+', ', ', result_address)  # remove zipcode multiple `,`
    result_address = remove_alone_char(result_address)  # remove `-`, `.`, `,` if they stand alone
    result_address = re.sub(' +', ' ', result_address)
    return result_address.strip()


class Inspectorio:
    @classmethod
    def load_data(cls, data_path, batch_size, vocab):
        df = pd.read_excel(data_path)
        df['content'] = df.apply(lambda row: process_helper(row['address']), axis=1)
        data, data_length = load_padded_data(df, vocab)
        data = np.array(data)
        origin_data = df[['name', 'address']].values
        data_dict = {"data": [], "length": [], "raw": []}

        for idx in tqdm(
                range(0, len(data)),
                desc="Load data"
        ):
            data_dict["data"].append(data[idx])
            data_dict["length"].append(data_length[idx])
            data_dict["raw"].append(idx)

        array = np.array(data_dict["data"])
        lengths = np.array(data_dict["length"])
        raw_array = np.array(data_dict["raw"])

        data = TensorDataset(
            torch.from_numpy(array).type(torch.LongTensor),
            torch.ByteTensor(lengths),
            torch.from_numpy(raw_array)
        )
        return DataLoader(data, batch_size=batch_size), origin_data

