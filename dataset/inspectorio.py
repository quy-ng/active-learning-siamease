import re
import pandas as pd
from sklearn.utils import shuffle
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
        self.data = shuffle(data.values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feed = self.data[index]
        return feed


if __name__ == '__main__':
    from dataset.augmentation import augment

    dataset = Inspectorio('~/Desktop/active_learning_data.xlsx', transform=augment)
    # for i in range(len(foo)):
    #     sample = foo[i]
    #     print(sample)
    #     break
    # print('#' * 10)
    print(dataset[0:3])
    print(dataset[2])

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2)
    for batch_idx, batch in enumerate(dataloader):
        print(batch_idx, batch)
