import re
import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import Dataset

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
    result_address = row[1]
    result_address = result_address.lower()
    result_address = re.sub('null', ' ', result_address)
    result_address = re.sub(r'(0)\1{4}', ' ', result_address)  # remove zipcode 00000
    result_address = re.sub(r'(,)\1+', ', ', result_address)  # remove zipcode multiple `,`
    result_address = remove_alone_char(result_address)  # remove `-`, `.`, `,` if they stand alone
    result_address = re.sub(' +', ' ', result_address)
    return row[0].lower(), result_address.strip()


class Inspectorio(Dataset):

    def __init__(self, file_path, transform=None):
        df = pd.read_excel(file_path)
        if transform is not None:
            augmented_df = transform(df)
            data = augmented_df.apply(lambda row: process_helper(row), axis=1)
        else:
            data = df[['name', 'address']].apply(lambda row: process_helper(row), axis=1)
        self.data = shuffle(data.values)
        length_max = 0
        for i in self.data:
            _len = len(i[0]) + len(i[1]) + 2  # 2 for join name and address
            if _len > length_max:
                length_max = _len
        self.length_max = length_max

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feed = self.data[index]
        return feed


if __name__ == '__main__':
    from dataset.augmentation import augment_dataframe

    dataset = Inspectorio('~/Desktop/active_learning_data.xlsx', transform=augment_dataframe)
    print(dataset[0:3])
    print(dataset[2])

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2)
    for batch_idx, batch in enumerate(dataloader):
        print(batch_idx, batch)
        break

    print('#' * 10)
    dataset = Inspectorio('~/Desktop/active_learning_data.xlsx', transform=None)
    print(dataset[0:3])

    print('#' * 10)
    print(process_helper(('AL-KARAM TEXTILE MILLS (PVT) LIMITED - UNIT III', 'HT-11/1, LANDHI INDUSTRIAL AREA')))
    print(process_helper(('L&T GROUP COMPANY LIMITED',
                          '41/7 Tan Thoi Nhat 8 Street, Tan Thoi Nhat Ward,, District 12, Ho Chi Minh City, Vietnam, '
                          'Ho Chi Minh, VN 70000')))

    print(process_helper(("krishna beads industries llp", " - sector 63 no -")))
    print(process_helper(("h. s. craft manufacturing cop.", " , no - rd")))
