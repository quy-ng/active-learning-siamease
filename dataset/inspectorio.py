import re
import pandas as pd
from torch.utils.data import Dataset

__all__ = ['Inspectorio']


def process_helper(row):
    result = row[0] + ', ' + row[1]
    result = result.lower()
    result = re.sub('null', '', result)
    return result


class Inspectorio(Dataset):

    def __init__(self, file_path, transform=None):
        df = pd.read_excel(file_path)
        augmented_df = transform(df)
        data = augmented_df.apply(lambda row: process_helper(row), axis=1)
        self.data = data.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feed = self.data[index]
        if type(feed) is str:
            return [feed]
        return feed.tolist()


if __name__ == '__main__':
    from dataset.augmentation import augment

    foo = Inspectorio('~/Desktop/active_learning_data.xlsx', transform=augment)
    for i in range(len(foo)):
        sample = foo[i]
        print(sample)
        break
    print('#' * 10)
    print(foo[1:3])
